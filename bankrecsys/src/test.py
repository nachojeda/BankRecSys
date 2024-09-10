import numpy as np

class Test:
    def __init__(self):
        """
        Initialize the Model object.

        """
    def test_als_user_id(self, model, user_id, user_item_matrix, TOP_K ):
        """
        Test Alternating Least Square model on single user id

        :param model: ALS model
        :param user_id: User ID
        :param user_item_matrix: User-item matrix
        :param TOP_K: Number of top K results to recommend

        :return: List of top K recommended items
        """
        recommendations = model.recommend(user_id, user_item_matrix[user_id], N=TOP_K)[0]

        return recommendations
    
    def test_als_batch(self, model, df_test, user_id_col, user_item_matrix, TOP_K,):
        """
        Test Alternating Least Square model on batch of users

        :param model: ALS model
        :param df_test: Dataframe with test data
        :param user_id_col: User ID column name
        :param user_item_matrix: User-item matrix
        :param TOP_K: Number of top K results to recommend

        :return: List of top K recommended items
        """
        userids = np.arange(len(df_test[user_id_col].unique()))
        
        # I'm predicting more than 7 to have room of deleting products from last date and still have at least 7 recommended products
        ids, scores = model.recommend(userids, user_item_matrix[userids], N=TOP_K)

        return ids, scores
    
    def decode_integers_to_categorical(self, arr, mapping):
        """
        Decode integer values to categorical from mapping

        :param arr: List of integer values
        :param mapping: Mappping dict with values and categories

        :return: List of string categories
        """
        results = []
        for item in arr:
            x = mapping.get(item)
            results.append(x)

        return results

    def decode_integers_to_categorical_batch(self, arr, mapping):
        """
        Decode integer values to categorical from mapping

        :param arr: List of integer values
        :param mapping: Mappping dict with values and categories

        :return: List of string categories
        """
        return [[mapping.get(item) for item in array] for array in arr]

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
