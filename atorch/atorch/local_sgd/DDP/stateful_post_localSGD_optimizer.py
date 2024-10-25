from torch.distributed.optim.post_localSGD_optimizer import PostLocalSGDOptimizer

from .outer_optim_model_averager import OuterOptimPeriodicModelAverager


class StatefulPostLocalSGDOptimizer(PostLocalSGDOptimizer):
    def state_dict(self):
        post_local_sgd_sd = super().state_dict()
        if isinstance(self.averager, OuterOptimPeriodicModelAverager):
            averager_sd = self.averager.state_dict()
            post_local_sgd_sd["averager"] = averager_sd

        return post_local_sgd_sd

    def load_state_dict(self, state_dict):
        averager_sd = state_dict.pop("averager", None)
        if averager_sd is not None and isinstance(self.averager, OuterOptimPeriodicModelAverager):
            self.averager.load_state_dict(averager_sd)

        super().load_state_dict(state_dict)
