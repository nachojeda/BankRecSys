import argparse
import yaml
import os
import logging

from .preprocess import *

# Initialize logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define log message format
    handlers=[
        logging.StreamHandler()  # Logs to the console; you can add FileHandler to log to a file as well
    ]
)
logger = logging.getLogger(__name__)  # Get a logger instance

parser = argparse.ArgumentParser(description="Run an AI/ML job from YAML/JSON configs.")
parser.add_argument("files", nargs="*", help="Config files for the job (local path only).")
parser.add_argument("-e", "--extras", nargs="*", default=[], help="Additional config strings for the job.")
parser.add_argument("-s", "--schema", action="store_true", help="Print settings schema and exit.")

def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    if args.schema:
        # Print the schema of the settings
        logger.info("Schema details here...")
        return 0
    # Execute the main application logic here...

   # Ensure at least one file is provided
    if not args.files:
        logger.info("Error: No config files provided.")
        return 1

    # Load the first configuration file (assuming one config file is passed)
    config_file_path = args.files[0]

    logger.info(f"Looking for config file at: {os.path.abspath(config_file_path)}")

    with open(config_file_path, 'r') as conf:
        config = yaml.safe_load(conf)

    train_file = config["paths"]["train_file"]
    test_file = config["paths"]["test_file"]
    top_k = config["params"]["top_k"]
    seed = config["params"]["seed"]

    # Preprocessing 
    loader = Load(train_file)
    df = loader.read_data()
    logger.info(df.head())

    df_train = Preprocess(df)

    # I should include more preprocessing...
    # df_train.scale_features()
    
    df_train = df_train.select_registers(column_name='indfall', value="N")
    df_train = df_train.one_hot_to_labels(start_idx=24, new_col_name='financial_products')
    df_train = df_train.remove_nulls_from_column(column_name='')
    logger.info((df_train.head()))



    return 0

if __name__ == "__main__":
    main()

