import pandas as pd

class Submit:
    def __init__(self):
        """
        Initialize the Model object.

        """
    def submission(self, df_test, user_id_col, decoded_recommendations):
        """
        Make submission for Kaggle competition

        :param df_test:
        :param df_test:
        :param user_id:
        :decoded_recommendations:

        :results: Dataframe with submission
        """
        # Flatten each sublist into a single string
        reco_items_list = [' '.join(sublist) for sublist in decoded_recommendations]

        col_clients_ids = df_test[user_id_col].unique()

        # Ensure the length matches
        if len(reco_items_list) == len(col_clients_ids):
            # Create a new DataFrame
            df_submission = pd.DataFrame({
                'ncodpers': col_clients_ids,
                'added_products': reco_items_list
            })
        else:
            print("The flattened list does not match the length of the DataFrame.")
            print(len(reco_items_list))
            print(len(col_clients_ids))

        return df_submission
