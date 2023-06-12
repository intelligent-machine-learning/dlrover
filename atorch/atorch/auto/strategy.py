import pickle


class Strategy(object):
    def __init__(self, opt_list=None):
        """opt_list is a list of optimizations, each list item is
        a tuple of (optimization_method, config, tunable)
        """
        self.opt_list = [] if opt_list is None else opt_list

    def __len__(self):
        return len(self.opt_list)

    def __getitem__(self, idx):
        return self.opt_list[idx]

    def __setitem__(self, idx, value):
        self.opt_list[idx] = value

    def add_opt(self, opt):
        self.opt_list.append(opt)

    def is_tunable(self):
        for item in self.opt_list:
            if item[2]:
                return True
        return False

    def get_parallel_mode(self):
        # return if_found, parallel_mode_config
        for item in self.opt_list:
            if item[0] == "parallel_mode":
                return True, item[1]
        return False, None

    def reset_config(self):
        # reset config and set tunable to True (except "parallel_mode")
        for idx, item in enumerate(self.opt_list):
            if item[1] is not None and item[0] != "parallel_mode":
                self.opt_list[idx] = (item[0], None, True)

    def set_tunable_value(self, opt_lib):
        # set tunable to proper value if it is None
        for idx, item in enumerate(self.opt_list):
            if item[2] is None:
                if item[1] is not None:
                    # config is set, non-tunable.
                    tunable = False
                else:
                    # same as corresponding opt method.
                    tunable = opt_lib[item[0]].is_tunable
                self.opt_list[idx] = (item[0], item[1], tunable)

    def adjust_data_parallel(self, data_parallel_size):
        for idx, item in enumerate(self.opt_list):
            if item[0] == "parallel_mode":
                if item[1] is not None:
                    # item[1] is tuple(list(name, size), optional(rank_list))
                    pg_list = []
                    for name, size in item[1][0]:
                        if name == "data":
                            pg_list.append((name, data_parallel_size))
                        else:
                            pg_list.append((name, size))
                    new_config = (pg_list, item[1][1])
                    self.opt_list[idx] = (item[0], new_config, False)
                else:
                    self.opt_list[idx] = (item[0], ([("data", data_parallel_size)], None), False)
                break

    def remove_distributed_method(self, opt_lib):
        # return list of removed distributed opt methods or None if nothing removed.
        opt_to_remove = []
        for item in self.opt_list:
            if opt_lib[item[0]].distributed_only(item[1]):
                opt_to_remove.append(item)
        if len(opt_to_remove) > 0:
            removed_names = [item[0] for item in opt_to_remove]
            for item in opt_to_remove:
                self.opt_list.remove(item)
            return removed_names
        else:
            return None

    def convert_strategy_to_easydl_format(self):
        edl_strategy = []
        for item in self.opt_list:
            edl_strategy.append((item[0], pickle.dumps(item[1]), item[2]))
        return edl_strategy

    def __str__(self):
        """output names of all optimization methods.
        For "parallel_mode", also show config.
        """
        output = f"Acceleration strategy has {len(self)} optimization methods"
        for idx, item in enumerate(self.opt_list):
            output += f"\n[{idx}] {item[0]}"
            if item[0] == "parallel_mode" and item[1] is not None:
                output += f" : {item[1][0]}"
        return output

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for idx, item in enumerate(self.opt_list):
            if item != other[idx]:
                return False
        return True
