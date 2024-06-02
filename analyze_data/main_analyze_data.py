import argparse

from analyze_data import AnalyzeDataDB1
from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)
    analyze = AnalyzeDataDB1(amount_max=10, **config["DB"])
    dico_result = analyze.analyze_results()
    for sport, dico in dico_result.items():
        print(f"\nSport: {sport}")
        for key, value in dico.items():
            print(f"{key}: {value}")
    analyze.clear_folder()
    analyze.plot_results()
    analyze.close_engine()

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Script with config and env file paths."
    )
    parser.add_argument(
        "--config_path",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env_path",
        default=".env.example",
        help="Path to the environment file (default: .env.example)",
    )

    args = parser.parse_args()
    config = load_config(args.config_path, args.env_path)

    main(args.config_path, args.env_path)

    # python3 analyze_data/main_analyze_data.py --config_path config/config_gagou.yaml --env_path config/.env.gagou