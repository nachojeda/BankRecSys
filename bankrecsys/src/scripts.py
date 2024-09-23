import argparse
import yaml
import os
import logging

from .preprocess import *
from .model import *
from .test import *

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
    als_params = config["als_params"]
    top_k = config["metrics"]["top_k"]

    # Preprocessing 
    preprocessor = Preprocess(train_file, nrows=100)
    preprocessor.read_data()

    # I should include more preprocessing...
    # df_train.scale_features()
    preprocessor.select_registers(column_name='indfall', value="N")
    preprocessor.one_hot_to_labels(start_idx=24, new_col_name='financial_products')
    preprocessor.remove_nulls_from_column(column_name='financial_products')
    preprocessor.timestamp_to_days_elapsed_weighted(timestamp_col_name="fecha_dato")
    df_train_processed, mapping = preprocessor.encode_categorical_to_integers(column_name='financial_products')
    user_item_matrix = preprocessor.user_item_matrix(weight_column="weight", user_id="ncodpers", item_id="financial_products_encoded")

    logger.info(df_train_processed.head())
    # logger.info(mapping)
    # logger.info(user_item_matrix)
    logger.info(type(user_item_matrix))
    logger.info("Preprocessed ✅")

    # Training
    model = Model(user_item_matrix=user_item_matrix, params=als_params)
    fitted_model = model.als_train()
    # model.save(path="../models")
    logger.info("Model trained ✅")

    # Testing
    test = Test(model=fitted_model, data_path=test_file, user_id_col="ncodpers", user_item_matrix=user_item_matrix, TOP_K=top_k)
    test.test_als_batch()
    results = test.decode_integers_to_categorical_batch(mapping)
    # logger.info(results)
    logger.info("Model tested ✅")

    # Submission
    df_submission = test.submission()
    logger.info(df_submission.head())
    
    logger.info("Done!")
    return 0

if __name__ == "__main__":
    main()

