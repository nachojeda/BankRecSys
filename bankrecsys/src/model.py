import implicit
import pickle

# Move this to a config file
params={
    "factors": 100,
    "regularization": 0.1,
    "iterations": 50
}

class Model:
    def __init__(self):
        """
        Initialize the Model object.

        """
    
    def als_train(self, user_item_matrix, params):
        """
        Train Alternating Least Square model 

        :param user_item_matrix:
        :param params:

        :return:
        """
        model = implicit.als.AlternatingLeastSquares(params)
        model.fit(user_item_matrix)

        return model
    
    def save_model_pickle(self, path, model):
        """
        Save trained model into a pickle file

        :param model:

        :return: None
        """
        with open(path, "wb") as f:
            pickle.dump(model, f)