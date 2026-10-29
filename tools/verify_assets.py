"""Verify the external production bundle without loading the catalog into RAM."""

import argparse
from pathlib import Path
from product_search.artifacts import read_json, resolve_paths, verify_assets, verify_encoder_assets
import product_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_file = args.config.resolve()
    config = read_json(config_file)
    contract = read_json(Path(product_search.__file__).with_name("frozen_contract.json"))
    verify_assets(contract["assets"], resolve_paths(config["assets"], config_file.parent))
    encoder = resolve_paths({"policy": config["encoder_policy"]}, config_file.parent)["policy"]
    verify_encoder_assets(encoder)
    print("All production and encoder file hashes match the frozen contract.")


if __name__ == "__main__":
    main()
