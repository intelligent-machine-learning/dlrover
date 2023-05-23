# coding=utf-8
from __future__ import absolute_import, unicode_literals

from argparse import ArgumentParser
from string import Template

if __name__ == "__main__":
    # Running in atorch root dir
    parser = ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    with open("setup.py.tpl", encoding="u8") as fin, open("setup.py", "w", encoding="u8") as fout:
        t = Template(fin.read())
        fout.write(t.safe_substitute(version=args.version))
