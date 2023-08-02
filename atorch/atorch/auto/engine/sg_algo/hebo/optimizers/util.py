from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace


def parse_space_from_bayesmark(api_config) -> DesignSpace:
    """
    Parse design space of bayesmark (https://github.com/uber/bayesmark)
    """
    space = DesignSpace()
    params = []
    for param_name in api_config:
        param_conf = api_config[param_name]
        param_type = param_conf["type"]
        param_space = param_conf.get("space", None)
        param_range = param_conf.get("range", None)
        param_values = param_conf.get("values", None)

        bo_param_conf = {"name": param_name}
        if param_type == "int":
            bo_param_conf["type"] = "int"
            bo_param_conf["lb"] = param_range[0]
            bo_param_conf["ub"] = param_range[1]
        elif param_type == "bool":
            bo_param_conf["type"] = "bool"
        elif param_type in ("cat", "ordinal"):
            bo_param_conf["type"] = "cat"
            bo_param_conf["categories"] = list(set(param_values))
        elif param_type == "real":
            if param_space in ("log", "logit"):
                bo_param_conf["type"] = "pow"
                bo_param_conf["base"] = 10
                bo_param_conf["lb"] = param_range[0]
                bo_param_conf["ub"] = param_range[1]
            else:
                bo_param_conf["type"] = "num"
                bo_param_conf["lb"] = param_range[0]
                bo_param_conf["ub"] = param_range[1]
        else:
            assert False, "type %s not handled in API" % param_type
        params.append(bo_param_conf)
    space.parse(params)
    return space
