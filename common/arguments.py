import argparse
from pathlib import Path

import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser("Tooth Step Simulation")

    parser.add_argument("--config", type=str, default=Path(__file__).parent.parent / "config" / "config.yaml")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    args.n_actions = args.width * args.height
    args.save_path = Path(__file__).parent.parent / "models"
    args.device = torch.device("cuda" if torch.cuda else "cpu")
    return args
