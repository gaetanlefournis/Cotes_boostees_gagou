import argparse

from retrieve_data_wepari.retriever_we_pari_psel import RetrieverPSELWePari
from retrieve_data_wepari.retriever_we_pari_winamax import \
    RetrieverWinamaxWePari
from utils.tools import load_config


def main(config_path, env_path):
    config = load_config(config_path, env_path)
    list_websites = {"winamax": RetrieverWinamaxWePari, "PSEL": RetrieverPSELWePari}
    for site in list_websites.keys():
        print(f"\nRetrieving data for {site} :")
        retriever_wepari = list_websites[site](**config["DB"], global_retrieve=config["SPECIFIC"]["global_retrieve"], table=site)
        retriever_wepari()


if __name__ == "__main__":
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
    main(args.config_path, args.env_path)

    # python3 retrieve_data_wepari/main_retrieve.py --config_path config/config_gagou.yaml --env_path config/.env.gagou