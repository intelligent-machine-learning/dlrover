from deepspeed.utils import log_dist


# helper function to map between DS policies and DS containers
def policy_to_ds_container(**kwargs):

    from .containers import DS_LLAMAContainer, LLAMALayerPolicy

    policy_to_container = {
        LLAMALayerPolicy: DS_LLAMAContainer,
    }

    container = None
    policy = kwargs["policy"]
    assert policy is not None, "Policy cannot be None"
    policy_type = type(policy)

    if policy_type not in policy_to_container:
        log_dist(f"Policy type {policy_type} not supported", [0])
    else:
        container = policy_to_container[policy_type](**kwargs)

    return container
