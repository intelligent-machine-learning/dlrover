import argparse
import importlib
import os


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparser_dir = "config_subparsers"
    supported_subparsers = [
        f.split(".")[0] for f in os.listdir(subparser_dir) if f.endswith(".py") and f != "__init__.py"
    ]
    target_modules = {}

    for sub in supported_subparsers:
        module = importlib.import_module(f"{subparser_dir}.{sub}")
        subparser = subparsers.add_parser(sub)
        module.BuildRender.add_arguments(subparser)
        target_modules[sub] = module

    args = parser.parse_args()
    if args.cache:
        return

    if args.subcommand in supported_subparsers:
        non_target = [i for i in supported_subparsers if i != args.subcommand]
        args.non_target = non_target
        module = target_modules[args.subcommand]
        obj = module.BuildRender(args)
        obj.run()


if __name__ == "__main__":
    main()
