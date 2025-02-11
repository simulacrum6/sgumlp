from argparse import ArgumentParser
from pathlib import Path

from .experiments import cv_experiment, ood_experiment
from .data import download_benchmark_datasets

def get_data_download_parser(subparsers):
    parser = subparsers.add_parser('download')
    parser.add_argument("--out_dir", type=str, default="data",
                       help="Path to the root directory of the datasets")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing datasets", default=False)
    return parser

def get_experiment_parser(subparsers):
    parser = subparsers.add_parser('experiment')
    parser.add_argument("name", choices=["replication"])
    parser.add_argument("--root_dir", type=str, default="data/config")
    return parser

def get_run_parser(subparsers):
    parser = subparsers.add_parser('run')
    parser.add_argument("type", choices=["cv", "ood"])
    parser.add_argument("cfg_path", type=str)
    return parser

if __name__ == '__main__':
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command', required=True)
    get_data_download_parser(subparsers)
    get_experiment_parser(subparsers)
    get_run_parser(subparsers)

    args = parser.parse_args()

    if args.command == "download":
        download_benchmark_datasets(args.out_dir, overwrite=args.overwrite)

    if args.command == "experiment":
        cfg_path = Path(args.root_dir) / f"{args.name}.experiment.json"
        cv_experiment(str(cfg_path))

    if args.command == "run":
        match args.type:
            case "cv":
                cv_experiment(args.cfg_path)
            case "ood":
                ood_experiment(args.cfg_path)