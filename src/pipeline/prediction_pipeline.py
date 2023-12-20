import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd



class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred



        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 RowNumber:float,
                 CustomerId:float,
                 Surname:str,
                 CreditScore:float,
                 Geography:str,
                 Gender:str,
                 Age:int,
                 Tenure:int,
                 Balance:int,
                 NumOfProducts:int,
                 HasCrCard:str,
                 IsActiveMember:str,
                 EstimatedSalary:int,
                 Exited:int):
        
        self.RowNumber=RowNumber
        self.CustomerId=CustomerId
        self.Surname=Surname
        self.CreditScore=CreditScore
        self.Geography=Geography
        self.Gender=Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts=NumOfProducts
        self.HasCrCard=HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary
        self.Exited = Exited



    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'RowNumber':[self.RowNumber],
                'CustomerId':[self.CustomerId],
                'Surname':[self.Surname],
                'CreditScore':[self.CreditScore],
                'Geography':[self.Geography],
                'Gender':[self.Gender],
                'Age':[self.Age],
                'Tenure':[self.Tenure],
                'Balance':[self.Balance],
                'NumOfProducts':[self.NumOfProducts],
                'IsActiveMember':[self.IsActiveMember],
                'HasCrCard':[self.HasCrCard],
                'EstimatedSalary':[self.EstimatedSalary],
                'Exited':[self.Exited]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

































