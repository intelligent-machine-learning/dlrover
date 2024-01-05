import configparser

config = configparser.ConfigParser()

config["M"] = {
    "ENABLE": "True",
    "THRESHOLD": "4096",
    "BITS": "4",
    "SCALE_TYPE": "group",
    "QUANT_TYPE": "nonlinear",
    "ROUND_TYPE": "real-nearest",
    "GROUP_SIZE": "128",
    "SIGNED": "True",
}

config["SQM"] = {
    "ENABLE": "True",
    "THRESHOLD": "4096",
    "BITS": "4",
    "SCALE_TYPE": "rank1",
    "QUANT_TYPE": "power-1",
    "ROUND_TYPE": "real-nearest",
    "GROUP_SIZE": "128",
    "SIGNED": "False",
}


def get_config(q_bits):
    config["M"]["BITS"] = str(q_bits)
    config["SQM"]["BITS"] = str(q_bits)

    return config
