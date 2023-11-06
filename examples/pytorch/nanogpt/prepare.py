# Copyright 2023 The DLRover Authors. All rights reserved.
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
import os
import pickle

import numpy as np


def prepare_dataset(input_file_path, output_dir):
    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # encoder: take a string, output a list of integers
    def encode(string):
        return [stoi[c] for c in string]

    # decoder: take a list of integers, output a string
    def decode(list):
        return "".join([itos[i] for i in list])

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]  # noqa

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    if not output_dir:
        output_dir = os.path.dirname(input_file_path)
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the tiny dataset.")
    parser.add_argument(
        "--src_data_path", type=str, help="Path to the source data file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Path to output the result."
    )

    args = parser.parse_args()
    src_data_path = args.src_data_path
    output_dir = args.output_dir
    print(f"src_data_path: {src_data_path}, output_dir: {output_dir}")

    if not os.path.exists(src_data_path):
        print(f"Error: The file '{src_data_path}' does not exist.")
        exit(1)

    prepare_dataset(src_data_path, output_dir)
