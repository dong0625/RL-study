import argparse
from omegaconf import OmegaConf

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config/default.yaml')

    args, unknown = parser.parse_known_args()

    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(unknown)

    merged_conf = OmegaConf.merge(conf, cli_conf)
    
    return merged_conf