import implicit
import pickle

class Model:
    def __init__(self, user_item_matrix, params):
        """
        Initialize the Model object.

        """
        self.user_item_matrix = user_item_matrix  # User item matrix used in RecoSys
        self.params = params  # Model parameters
    
    def als_train(self):
        """
        Train Alternating Least Square model 

        :param user_item_matrix:
        :param params:

        :return:
        """
        model = implicit.als.AlternatingLeastSquares(**self.params)
        model.fit(self.user_item_matrix)
        self.model = model
        
        return model
    
    def save(self, path):
        """
        Save trained model into a pickle file

        :param model:

        :return: None
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)