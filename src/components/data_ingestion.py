import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# dataclass for storing variables
# initializing data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

#creating a class for data ingestion 
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started")

        try:
            df=pd.read_csv(os.path.join("notebook/data","Churn_Modelling.csv"))
            logging.info("dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info("Train Test Split")

            train_set,test_set=train_test_split(df,test_size=0.33,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("ingestion of data completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured at data ingestion state")
            raise CustomException(e,sys)
        

# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     obj_data_transformation=DataTransformation()
#     train_arr,test_arr,_=obj_data_transformation.initiate_data_transformation(train_data_path,test_data_path)




































