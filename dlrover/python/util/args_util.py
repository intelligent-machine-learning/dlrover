# Copyright 2025 The EasyDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ast


def pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise argparse.ArgumentTypeError(
            f"Positive integer argument required, but got {res}"
        )
    return res


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "t", "y", "1"}:
        return True
    elif value.lower() in {"false", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_tuple_list(value):
    """
    Format: [(${value1}, ${value2}, ${value3}), ...]
    Support tuple2 and tuple3.
    """
    if not value or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and all(
            isinstance(t, tuple) and (len(t) == 2 or len(t) == 3)
            for t in parsed
        ):
            return parsed
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected format: [(v1, v2), ...]"
        )


def parse_dict(value):
    """
    Format：{${key}: ${value}, ...}
    """
    if not value or value == "":
        return {}
    try:
        parsed_dict = ast.literal_eval(value)

        if isinstance(parsed_dict, dict):
            return parsed_dict
        else:
            raise ValueError("invalid format: not dict")
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected format: {k1: v1, ...}"
        )


def parse_tuple_dict(value):
    """
    Format：{(${value1}, ${value2}): ${boolean}, ...}
    Support tuple2.
    """
    if not value or value == "":
        return {}
    try:
        result_dict = {}
        parsed_dict = ast.literal_eval(value)

        if isinstance(parsed_dict, dict):
            for key, value in parsed_dict.items():
                if not isinstance(key, tuple):
                    raise ValueError("invalid format: key should be tuple")

                if isinstance(value, bool):
                    result_dict[key] = value
                elif isinstance(value, str):
                    result_dict[key] = str2bool(value)
                else:
                    raise ValueError(
                        "invalid format: value should be boolean "
                        "or boolean expression in str"
                    )
            return result_dict
        else:
            raise ValueError("invalid format: not dict")
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected format: {(v1, v2): ${boolean}, ...}"
        )
