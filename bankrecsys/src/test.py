import numpy as np
import pandas as pd

class Test:
    def __init__(self, model, data_path, user_id_col, user_item_matrix, TOP_K):
        """
        Initialize the Model object.

        """
        self.model = model
        self.user_id_col = user_id_col
        self.user_item_matrix = user_item_matrix
        self.top_k = TOP_K

        self.df_test = pd.read_csv(data_path)

    # def test_als_user_id(self):
    #     """
    #     Test Alternating Least Square model on single user id

    #     :param model: ALS model
    #     :param user_id: User ID
    #     :param user_item_matrix: User-item matrix
    #     :param TOP_K: Number of top K results to recommend

    #     :return: List of top K recommended items
    #     """
    #     recommendations = self.model.recommend(self.user_id, self.user_item_matrix[self.user_id], N=self.top_k)[0]

    #     return recommendations
    
    def test_als_batch(self):
        """
        Test Alternating Least Square model on batch of users

        :param model: ALS model
        :param df_test: Dataframe with test data
        :param user_id_col: User ID column name
        :param user_item_matrix: User-item matrix
        :param TOP_K: Number of top K results to recommend

        :return: List of top K recommended items
        """
        userids = np.arange(len(self.df_test[self.user_id_col].unique()))
        
        # I'm predicting more than 7 to have room of deleting products from last date and still have at least 7 recommended products
        self.ids, scores = self.model.recommend(userids, self.user_item_matrix[userids], N=self.top_k)

        return self.ids, scores
    
    # def decode_integers_to_categorical(self, mapping):
    #     """
    #     Decode integer values to categorical from mapping

    #     :param arr: List of integer values
    #     :param mapping: Mappping dict with values and categories

    #     :return: List of string categories
    #     """
    #     results = []
    #     for item in self.ids:
    #         x = mapping.get(item)
    #         results.append(x)

    #     return results

    def decode_integers_to_categorical_batch(self, mapping):
        """
        Decode integer values to categorical from mapping

        :param arr: List of integer values
        :param mapping: Mappping dict with values and categories

        :return: List of string categories
        """
        return [[mapping.get(item) for item in array] for array in self.ids]

    def remove_current_items(self, row, added_items, items):
        """
        Remove current items aquired by user from recommendations

        :param row:
        :param added_items:
        :param items:

        :return: Row with substracted items
        """
        added_products = set(row[added_items].split())
        financial_products = set(row[items].split())
        
        remaining_products = added_products - financial_products
        
        return ' '.join(remaining_products)
