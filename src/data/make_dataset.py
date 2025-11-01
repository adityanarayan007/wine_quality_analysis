#importing Libraries..
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split 


#Setup logging..
logging.basicConfig(level=logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')


#configure file Paths and values
RAW_DATA_PATH = "data/raw/winequality-red.csv"
PROCESSED_DATA_PATH = "data/processed/"
TRAIN_PATH = os.path.join(PROCESSED_DATA_PATH,"train.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_PATH,"test.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


#Loading the data...
def load_data( path:str )->pd.DataFrame:
    logging.info(f"Loading Data from {path}")
    df = pd.read_csv(path,sep=";")
    logging.info(f"raw Data Loaded with shape : {df.shape}")
    return df

#Basic Cleaing is not required as the dataset is clean and doesn't contain any null values

#Splitting the data into train and test
def split_data( df:pd.DataFrame, test_size = TEST_SIZE, random_state = RANDOM_STATE):
    logging.info(f"Splitting the data into 80-20, train-test size.")
    train,test = train_test_split(df,test_size=test_size,random_state=random_state)
    return train,test

#Saving the split and semi processed data
def save_data(df:pd.DataFrame, path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved data into {path}")

#Main Processing functon
def prepare_datset():
    df = load_data(RAW_DATA_PATH)
    train,test = split_data(df=df,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    save_data(df=train,path=TRAIN_PATH)
    save_data(df=test,path=TEST_PATH)
    logging.info("Dataset preparation complete..")

#Runner fucntion:
if __name__  == "__main__":
    prepare_datset()
