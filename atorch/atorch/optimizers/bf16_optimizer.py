# TODO:  Update overflow check + downscale to use Carl's fused kernel.
import torch
from fairscale.optim.oss import OSS
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.autograd import Variable


def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    """
    Copy model gradients to master gradients.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.
    """
    if flat_master:
        # The flattening may incur one more deep copy than is necessary.
        master_params[0].grad.data.copy_(_flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None


def master_params_to_model_params(model_params, master_params, flat_master=False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.
    """
    if flat_master:
        for model, master in zip(model_params, _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)


# based on https://github.com/THUDM/GLM/blob/main/fp16/fp16.py
class BF16Optimizer(torch.optim.Optimizer):
    """
    :class:`BF16Optimizer` is designed to wrap an existing PyTorch optimizer.
    """

    def __init__(self, init_optimizer, verbose=False):
        if not torch.cuda.is_available:
            raise SystemError("Cannot use bf16 without CUDA.")

        self.verbose = verbose

        self.optimizer = init_optimizer
        # init_state_dict sets up an alternative way to cast per-param state tensors.
        # Stashing here in case https://github.com/pytorch/pytorch/issues/7733 makes it necessary.
        # init_state_dict = init_optimizer.state_dict()

        self.fp16_groups = []
        self.fp32_from_fp16_groups = []
        self.fp32_from_fp32_groups = []

        for i, param_group in enumerate(
            self.optimizer.optim.param_groups if isinstance(self.optimizer, OSS) else self.optimizer.param_groups
        ):
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(param_group["params"]):
                if param.requires_grad:
                    if param.type() == "torch.cuda.HalfTensor" or param.type() == "torch.cuda.BFloat16Tensor":
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        param_group["params"][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                        # Reset existing state dict key to the new master param.
                        # We still need to recast per-param state tensors, if any, to FP32.
                        if param in self.optimizer.state:
                            self.optimizer.state[master_param] = self.optimizer.state.pop(param)
                    elif param.type() == "torch.cuda.FloatTensor":
                        fp32_params_this_group.append(param)
                        param_group["params"][i] = param
                    else:
                        raise TypeError(
                            "Wrapped parameters must be either "
                            "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                            "Received {}".format(param.type())
                        )

            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

    def __getstate__(self):
        raise RuntimeError("BF16FinetuneOptimizer should be serialized using state_dict().")

    def __setstate__(self, state):
        raise RuntimeError("BF16FinetuneOptimizer should be deserialized using load_state_dict().")

    def zero_grad(self, set_grads_to_None=False):
        """
        Zero fp32 and fp16 parameter grads.
        """
        # In principle, only the .grad attributes of the model params need to be zeroed,
        # because gradients are copied into the FP32 master params.  However, we zero
        # all gradients owned by the optimizer, just to be safe:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

        # Zero fp16 gradients owned by the model:
        for fp16_group in self.fp16_groups:
            for param in fp16_group:
                if set_grads_to_None:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach_()  # as in torch.optim.optimizer.zero_grad()
                        param.grad.zero_()

    def _master_params_to_model_params(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)

    def _model_params_to_master_params(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            master_params_to_model_params(fp32_from_fp16_group, fp16_group)

    # To consider:  Integrate distributed with this wrapper by registering a hook on each variable
    # that does the overflow check, gradient copy + downscale, and fp32 allreduce in a different stream.
    def _model_grads_to_master_grads(self):
        for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
            model_grads_to_master_grads(fp16_group, fp32_from_fp16_group)

    def clip_master_grads(self, max_norm, norm_type=2):
        """
        Clips fp32 master gradients via ``torch.nn.utils.clip_grad_norm``.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the current fp32 gradients (viewed as a single vector).

        .. warning::
        """
        if not self.overflow:
            fp32_params = []
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    fp32_params.append(param)
            return self.clip_grad_norm(fp32_params, max_norm, norm_type)
        else:
            return -1

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::

            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        state_dict["fp32_from_fp16"] = self.fp32_from_fp16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.

        Example::

            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.first_closure_call_this_step = state_dict["first_closure_call_this_step"]
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        for current_group, saved_group in zip(self.fp32_from_fp16_groups, state_dict["fp32_from_fp16"]):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def step(self, closure=None):  # could add clip option.

        self.update_master_grads()

        if closure is not None:
            retval = self._step_with_closure(closure)
        else:
            retval = self.optimizer.step()

        self._master_params_to_model_params()

        return retval

    def _step_with_closure(self, closure):
        def wrapped_closure():
            # helpful for debugging
            # print("Calling wrapped_closure, first_closure_call_this_step = {}"
            #       .format(self.first_closure_call_this_step))
            if self.first_closure_call_this_step:
                # We expect that the fp16 params are initially fresh on entering self.step(),
                # so _master_params_to_model_params() is unnecessary the first time wrapped_closure()
                # is called within self.optimizer.step().
                self.first_closure_call_this_step = False
            else:
                # If self.optimizer.step() internally calls wrapped_closure more than once,
                # it may update the fp32 params after each call.  However, self.optimizer
                # doesn't know about the fp16 params at all.  If the fp32 params get updated,
                # we can't rely on self.optimizer to refresh the fp16 params.  We need
                # to handle that manually:
                self._master_params_to_model_params()
            # Our API expects the user to give us ownership of the backward() call by
            # replacing all calls to loss.backward() with optimizer.backward(loss).
            # This requirement holds whether or not the call to backward() is made within a closure.
            # If the user is properly calling optimizer.backward(loss) within "closure,"
            # calling closure() here will give the fp32 master params fresh gradients
            # for the optimizer to play with, so all wrapped_closure needs to do is call
            # closure() and return the loss.
            temp_loss = closure()
            return temp_loss

        retval = self.optimizer.step(wrapped_closure)

        self.first_closure_call_this_step = True

        return retval

    def backward(self, loss, update_master_grads=True, retain_graph=False):
        if update_master_grads:
            self.update_master_grads()

    def update_master_grads(self):
        self._model_grads_to_master_grads()

    @property
    def param_groups(self):
        return self.optimizer.optim.param_groups if isinstance(self.optimizer, OSS) else self.optimizer.param_groups
