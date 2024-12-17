# Copyright 2024 The DLRover Authors. All rights reserved.
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
from pathlib import Path

from py_xpu_timer import hosting_service_pb2  # type: ignore[attr-defined]


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_stack = False
        self.ranks = set()

    def add_rank(self, rank):
        self.ranks.add(rank)


class StackTrie:
    def __init__(self, all_ranks):
        self.root = TrieNode()
        self.all_ranks = all_ranks

    def insert(self, words, rank):
        node = self.root
        for word in words:
            if "lto_priv" in word:
                break
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
            node.ranks.add(rank)
        node.is_end_of_stack = True
        node.add_rank(rank)

    def _format_rank_str(self, ranks):

        leak_ranks = list(self.all_ranks - set(ranks))
        ranks = list(ranks)

        def _inner_format(ranks):
            """fold continuous ranks, [0,1,2,5,6,7]->[0-2,5-7]
            return has stack and leak stack, suppose we have 8 ranks(0-7)
            [0,1,2,5,6,7]->0-2/5-7|3-4, means rank 0-2,5-7 has this stacktrace,
            while rank 3-4 do not have this stacktrace
            """
            ranks = sorted(ranks)
            str_buf = []
            low = 0
            high = 0
            total = len(ranks)
            while high < total - 1:
                low_value = ranks[low]
                high_value = ranks[high]
                while high < total - 1 and high_value + 1 == ranks[high + 1]:
                    high += 1
                    high_value = ranks[high]
                low = high + 1
                high += 1
                if low_value != high_value:
                    str_buf.append(f"{low_value}-{high_value}")
                else:
                    str_buf.append(str(low_value))
            if high == total - 1:
                str_buf.append(str(ranks[high]))
            return "/".join(str_buf)

        has_stack_ranks = _inner_format(ranks)
        leak_stack_ranks = _inner_format(leak_ranks)
        return f"@{'|'.join([has_stack_ranks, leak_stack_ranks])}"

    def _traverse_with_all_stack(self, node, path):
        for word, child in node.children.items():
            rank_str = self._format_rank_str(child.ranks)
            if child.is_end_of_stack:
                yield ";".join(path + [word]) + rank_str
            word += rank_str
            yield from self._traverse_with_all_stack(child, path + [word])

    def __iter__(self):
        yield from self._traverse_with_all_stack(self.root, [])


class StackViewer:
    def __init__(self, path):
        p = Path(path)
        self.path = path
        self.files = sorted(p.glob("*stacktrace"))
        if not self.files:
            print(f"no stacktrace files in {path}")
            exit(1)
        # files format is 00003-00008.stacktrace
        self.world_size = int(self.files[0].name[6:11])
        self.all_ranks = set(range(self.world_size))

        self._parse("cpp")
        self._parse("py")

    def _parse(self, mode):
        self.stack_trie = StackTrie(self.all_ranks)
        for f in self.files:
            self._parse_one(f, mode)
        with open(f"{self.path}/{mode}_stack", "w") as f:
            for stack in self.stack_trie:
                f.write(f"{stack} 1\n")
        os.system(
            "flamegraph.pl --color python --width 1600 --title "
            f"'merge stack in {mode}' < {self.path}/{mode}_stack "
            f"> {self.path}/{mode}_stack.svg"
        )

    def _frame_hash(self, stracetrace, rank):
        for i in stracetrace:
            buf = []
            for index, frame in enumerate(i.frames[::-1]):
                func_file_name = f"{frame.func_name}@{frame.file_name}"
                buf.append(func_file_name)
            self.stack_trie.insert(buf, rank)

    def _parse_one(self, path, mode):
        st = hosting_service_pb2.Stacktrace()
        # 00003-00008.stacktrace
        rank = int(path.name[:5])
        with open(path, "rb") as f:
            st.ParseFromString(f.read())
            if st.pstack_stderr:
                print(st.pstack_stderr)
        self.stack_trie.insert([f"State@{st.process_state}"], rank)
        if mode == "cpp":
            self._frame_hash(st.stacktrace, rank)
        else:
            self._frame_hash(st.py_stacktrace, rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="./")
    args = parser.parse_args()
    StackViewer(args.path)


if __name__ == "__main__":
    main()
