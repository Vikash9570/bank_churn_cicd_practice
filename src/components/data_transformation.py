import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import os
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import logging

from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformed_object(self):
        try:
            #defining column
            categorical_col=["Gender"]
            numerical_col=["CreditScore","Age","Tenure","Balance","NumOfProducts",
                           "HasCrCard","IsActiveMember","EstimatedSalary"]
            
            # defining gender
            Gender_catagories=["Male","Female","Other"]

            logging.info("pipeline initiated")
            # numerical pipeline 
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            ) 
            # categorical column
            cat_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                       ("ordinalencoder",OrdinalEncoder(categories=[Gender_catagories])),
                       ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_col),
                ("cat_pipeline",cat_pipeline,categorical_col)
            ])

            return preprocessor

            logging.info("pipeline completed")
            
        except Exception as e:
            logging.info("error in data DataTransformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # reading dataset

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info(f'checking shape of data{train_df.shape,test_df.shape}')

            logging.info("read train and test data completes")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtaining preprocessor object")
            preprocessing_obj=self.get_data_transformed_object()

            target_column_name="Exited"
            drop_columns=[target_column_name,"RowNumber","CustomerId","Surname","Geography"]

            logging.info("columns droped")
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            logging.info("applied preprocessing object on train and test data")

            # concatnating train and test arr
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessing file saved")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            logging.info("error occurs in data trainformation ")
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     obj_data_transformation=DataTransformation()
#     train_arr,test_arr,_=obj_data_transformation.initiate_data_transformation(train_data_path,test_data_path)










































