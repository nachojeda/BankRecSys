import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.impute import SimpleImputer

from scipy.sparse import csr_matrix

import pandas as pd

class Preprocess:
    def __init__(self, data_path, scaling_method=None, features=None, nrows=100):
        """
        Initialize the Preprocess object with a pandas DataFrame.
        
        :param df: Input DataFrame to be preprocessed
        """
        self.data_path = data_path  # Path to data
        self.scaling_method = scaling_method  # Normalization/Standardization
        self.features = features  # Specific features to include
        self.nrows = nrows

    def read_data(self) -> pd.DataFrame:
        """
        Reads the data from the provided file path and returns it as a DataFrame.

        :return: DataFrame containing the data.
        """
        try:
            self.df = pd.read_csv(filepath_or_buffer=self.data_path, nrows=self.nrows)
            return self.df
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
        except pd.errors.ParserError:
            print("Error while parsing the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def scale_features(self, method='standard'):
        """
        Scale numerical features using standard or min-max scaling.
        
        :param method: Scaling method, 'standard' or 'minmax'
        :return: Scaled DataFrame
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method: choose 'standard' or 'minmax'")
        
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        return self.df
    
    def one_hot_to_labels(self, start_idx, new_col_name):
        """
        Decode numerical one hot encoded columns to single column labels and delete one hot encoding columns

        :param star_idx: Index from which starts the columns
        :param new_col_name: Name of the new column containing the labels

        :return: DataFrame with new column containing labels
        """
        one_hot_columns = self.df.columns[start_idx:]
        
        self.df[new_col_name] = self.df[one_hot_columns].apply(
            lambda row: ' '.join([col for col, val in row.items() if val == 1]),
            axis=1
        )
        self.df = self.df.drop(columns=one_hot_columns)
    
        return self.df
    
    def select_registers(self, column_name, value):
        """
        Select registers containing indicated value

        :param column_name: Column name from the dataframe
        :param value: Column value

        :return: DataFrame with selected registers
        """ 
        self.df =  self.df[self.df[column_name] == value]

        return self.df

    def remove_nulls_from_column(self, column_name):
        """
        Select registers containing indicated value

        :param column_name: Column name from the dataframe

        :return: DataFrame with removed nulls from selected column
        """ 
        self.df =  self.df[self.df[column_name].notnull()]

        return self.df
    
    def encode_categorical_to_integers(self, column_name):
        """
        Encodes the categorical values in a specific column of a DataFrame to integers.
        
        :param df: pandas DataFrame
        :param column_name: The name of the column containing categorical values to encode
        
        :return df: DataFrame with the column encoded to integers
        :return: mapping: A dictionary mapping the original categorical values to integers
        """
        mapping = {category: idx for idx, category in enumerate(self.df[column_name].unique())}

        self.df[column_name + '_encoded'] = self.df[column_name].map(mapping)

        # Reverse mapping to get category-integer
        inv_map = {v: k for k, v in mapping.items()}
        
        return self.df, inv_map

    def timestamp_to_days_elapsed_weighted(self, timestamp_col_name):
        """
        Convert tiemstamp to days elapsed since first date and weighted value (higher if more recent, less if less recent)
        
        :param tiemstamp_col_name: Column with timestamp value

        :return: Dataframe with column of days elapsed since first and column weigthed by recency
        """
        self.df['days_elapsed'] = (pd.to_datetime(self.df[timestamp_col_name], format="%Y-%m-%d") - pd.to_datetime(self.df[timestamp_col_name], format="%Y-%m-%d").min()).dt.days

        self.df['weight'] = 1 / (1 + self.df['days_elapsed'])

        return self.df

    def user_item_matrix(self, weight_column, user_id, item_id):
        """
        Creates user-item matrix with feature weighted

        :param weight_column: 
        :param user_id:
        :param item_id
        """
        user_item_matrix = csr_matrix((self.df[weight_column], (self.df[user_id], self.df[item_id]))) #.T.tocsr()

        return user_item_matrix