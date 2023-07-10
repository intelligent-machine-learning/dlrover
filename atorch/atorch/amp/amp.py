from apex import amp


def initialize(
    model,
    optimizers=None,
    enabled=True,
    opt_level="O1",
    keep_batchnorm_fp32=None,
    loss_scale=None,
    min_loss_scale=None,
    max_loss_scale=16777216.0,
    **kwargs,
):
    """
    Wrap `apex.amp.initialize`.
    Initialize your models, optimizers, and the Torch tensor and functional
    namespace according to the chosen opt_level and overridden properties,
    if any.

    Args:
        models: Models to modify/cast.
        optimizers: Optimizers to modify/cast. REQUIRED for training, optional
            for inference.
        enabled: If False, renders all Amp calls no-ops, so your script should
            run as if Amp were not present.
        opt_level: Pure or mixed precision optimization level. Accepted values
            are “O0”, “O1”, “O2”, and “O3”, explained in detail above.
        keep_batchnorm_fp32: Optional property override. If passed as a
            string, must be the string “True” or “False”.
        loss_scale: Optional property override. If passed as a string, must be
            a string representing a number, e.g., “128.0”, or the string
            “dynamic”.
        min_loss_scale: Sets a floor for the loss scale values that can be
            chosen by dynamic loss scaling. The default value of None means
            that no floor is imposed. If dynamic loss scaling is not used,
            min_loss_scale is ignored.
        max_loss_scale: Sets a ceiling for the loss scale values that can be
            chosen by dynamic loss scaling. If dynamic loss scaling is not
            used, max_loss_scale is ignored.

    Returns:
        Model(s) and optimizer(s) modified according to the opt_level. If
        either the models or optimizers args were lists, the corresponding
        return value will also be a list.
    """
    return amp.initialize(
        model,
        optimizers=optimizers,
        enabled=enabled,
        opt_level=opt_level,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        loss_scale=loss_scale,
        min_loss_scale=min_loss_scale,
        max_loss_scale=max_loss_scale,
        **kwargs,
    )


def scale_loss(loss, optimizers, **kwargs):
    """
    Wrap `apex.amp.scale_loss`.

    On context manager entrance, creates scaled_loss = (loss.float())*current
    loss scale. scaled_loss is yielded so that the user can call
    scaled_loss.backward():
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    On context manager exit (if delay_unscale=False), the gradients are
    checked for infs/NaNs and unscaled, so that optimizer.step() can be
    called.
    """
    return amp.scale_loss(loss, optimizers, **kwargs)


def master_params(optimizer):
    """
    Wrap `apex.amp.master_params`.

    Generator expression that iterates over the params owned by optimizer.

    Returns:
        optimizer: An optimizer previously returned from amp.initialize.
    """
    yield from amp.master_params(optimizer)


def state_dict(destination=None):
    """
    Wrap `apex.amp.state_dict`.

    To properly save and load your amp training, amp.state_dict() contains
    all loss_scalers.
    """
    return amp.state_dict(destination=destination)


def load_state_dict(state_dict):
    """
    Wrap `apex.amp.load_state_dict`.
    """
    return amp.load_state_dict(state_dict)
