# from xflow.modelhub.domain.modelhub_model import ModelHubModel
import dataclasses
import os
import sys
from typing import Iterable, Tuple, Union

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType


class ArgsUtils:
    @staticmethod
    def try_parse_args(
        args="sys",
        arg_type: Union[DataClassType, Iterable[DataClassType]] = None,
        allow_extra_keys=False,
    ) -> Tuple:
        # if arg_type is None:
        #     arg_type = cls._support_arg_classes()
        # else:
        #     parser = HfArgumentParser(arg_type)
        parser = HfArgumentParser(arg_type)

        arg_type_length = 1 if dataclasses.is_dataclass(arg_type) else len(arg_type)  # type: ignore[arg-type]

        if args == "sys":
            try:
                sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
            except ValueError:
                pass

            args = sys.argv
            if len(args) == 2 and args[1].endswith(".json"):
                # If we pass only one argument to the script, and it's a path to a json file,
                # let's parse it to get our arguments.
                parsed_args = parser.parse_json_file(
                    json_file=os.path.abspath(sys.argv[1] or args),
                    allow_extra_keys=allow_extra_keys,
                )
            else:
                parsed_args = parser.parse_args_into_dataclasses(return_remaining_strings=allow_extra_keys)
        elif isinstance(args, list):
            parsed_args = parser.parse_args_into_dataclasses(args, return_remaining_strings=allow_extra_keys)
        elif isinstance(args, str) and args.endswith(".json"):
            parsed_args = parser.parse_json_file(json_file=os.path.abspath(args), allow_extra_keys=allow_extra_keys)
        elif isinstance(args, dict):
            parsed_args = parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
        elif dataclasses.is_dataclass(args):
            parsed_args = (args,)
        elif isinstance(args, tuple):
            parsed_args = args
        else:
            raise NotImplementedError(
                f"init args {args} is not a supported format, only shell args, json and dict"
                f"args are supported to parse into dataclass"
            )
        # if (not aware_not_recognized_args) and (not is_dataclass(parsed_args[-1])):
        #     logger.warning(f"Some args are not parsed into dataclass: {parsed_args[-1]}")
        if allow_extra_keys and len(parsed_args) < arg_type_length + 1:
            return parsed_args + (None,)
        return parsed_args
