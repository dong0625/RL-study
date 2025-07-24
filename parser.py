import argparse
import yaml

def parse() -> argparse.Namespace:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    def add_arguments_from_config(parser: argparse.ArgumentParser, config_dict: dict, parent_key: str = ''):
        for key, value in config_dict.items():
            full_key = key
            if isinstance(value, dict):
                add_arguments_from_config(parser, value, full_key)
            elif isinstance(value, bool):
                parser.add_argument(f'--{full_key}', action='store_true', default=value)
            else:
                parser.add_argument(f'--{full_key}', type=type(value), default=value)
    
    add_arguments_from_config(parser, config)

    args = parser.parse_args()

    return args