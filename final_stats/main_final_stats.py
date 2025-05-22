import argparse
import time

from final_stats.final_stats import FinalStats
from utils.tools import load_config


def main(config_path : str, env_path : str) -> None:
    """
    Main function to process the analyzing of boosted odds database
    """
    config = load_config(config_path, env_path)
    try:
        analyze = FinalStats(**config["DB_VPS"])
        analyze.update_db()
        time.sleep(5)
        total_amount_won_wina, total_amount_won_psel, total_amount_won_unibet = analyze.analyze_results()
        print("total winamax : " + str(total_amount_won_wina), "total PSEL : " + str(total_amount_won_psel), "total unibet : " + str(total_amount_won_unibet))
    except Exception as e:
        print(e)
    finally:
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

    # python3 final_stats/main_final_stats.py --config_path config/config_gagou.yaml --env_path config/.env.gagou
    # python3 -m final_stats.main_final_stats --config_path config/config_gagou.yaml --env_path config/.env.gagou