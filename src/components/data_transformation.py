## I'll do FE in this Data Transformation step

import pandas as pd 
import numpy as np 
import os, sys 
from dataclasses import dataclass
from sklearn.impute import SimpleImputer ## Handle Missing Values
from sklearn.preprocessing import StandardScaler ## Feature Scaling
from sklearn.preprocessing import OrdinalEncoder ## Ordinal Encoding
## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

# To make the pickle file
from src.utils import save_object

# 2 Steps to be done in this Data Transformation Process
# 1.) Data Transformation Config - pkl file containing FE code
# 2.) Data TransformationConfig Class

# Data Transformation Config
@dataclass
class DataTransformationConfig:
    # Mai yha FE/Preprocessed data ki pkl file ka path dunga
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    # preprocessor.pkl will be used in DataTransformation Config

# Data TransformationConfig Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # This method will form pkl file
    def get_data_transformation_config(self):
        try:
            logging.info("Data Transformation Initiated.")
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            ## Defining the custom ranking for each ordinal variables
            cut_categories = ['Good', 'Fair', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("Pipeline Initiated.")

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            ## Categorical Pipline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor 
            # preprocessor is my FE processor
            logging.info("Pipeline Completed")

        except Exception as e:
            logging.error("Error in Data Transformation: %s", e)
            raise CustomException(e, sys)

    # This method will initiate the data transformation process
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read tarin & test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train & test Data completed")
            logging.info(f'Train DataFrame head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame head: \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_config()

            target_col_name = 'price'
            drop_cols = [target_col_name, 'id']
# Delete the 2 above features from the I/P features

            # Dividing features into I/P and Dependent features
            input_feature_train_df = train_df.drop(columns=drop_cols, axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=drop_cols, axis=1)
            target_feature_test_df = test_df[target_col_name]

            # Apply the Transformation
            # I/P feature preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            ## Combining the preprocessed I/P features with the target feature
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # From here the pickle file will be created in the artifacts folder
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor Pickle is Created & Saved")

            return(
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)



