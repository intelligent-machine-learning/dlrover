import atorch

p_mode = ([("data", atorch.world_size())], None)
strategy = [("parallel_mode", p_mode), "amp_native", ("fsdp")]
