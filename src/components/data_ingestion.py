## in this step I'll read the data and split it into train and test data
## I'll return train & test data from this DATA INGESTION STEP.

import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## In this step I have to do 2 main things:--
# 1.) Initialize the data ingestion configuration
# data ingestion class ke liye jo bhi parameters chahiye, like read krliya data ab isko dave krna hai kahi
# toh jaha save krna hai uss location ka path yaha hoga
# 2.) create a data ingestion class -  respomsible for reading the data, doing train-test split

## Initialize the data ingestion configuration

@dataclass
# using @dataclass I donot have to make the constructor
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    # File name - 'train.csv' that will be saved in artifacts folder
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Create a data ingestion class
class DataIngestion:
    # While initializing the DataIngestion class I should get the data_ingestion_config parameters

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    # Ye kaam krne se train_data_path, test, raw wale saare parameters aa jayenge

    # Initiate Data Ingestion
    # In this step I'll read data, train-test split, saving of the updated files all this I'll do
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method Starts")

        try:
            # Read Data
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info("Dataset read as pandas Dataframe")

            # Suppose the artifacts folder doesn't exists , then we have to create it
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Saving the Readed data to raw.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Data saved in raw.csv")

            # TRAIN_TEST Split
            logging.info("Train-Test Split Started")
            train_set, test_set = train_test_split(df, test_size = 0.30, random_state = 42)
            logging.info("Train-Test Split Ended")

            # Saving
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)
            logging.info("Ingestion of Data is Completed!!!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error Occured in Data Ingestion Config")


## I'll run this code usinf the training pipeline