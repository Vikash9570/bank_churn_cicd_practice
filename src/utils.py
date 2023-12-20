import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score


from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report:dict = {}
        report_confusion_matrix:dict={}
        for i in range(len(models)):
            model = list(models.values())[i]
            # train models
            model.fit(X_train,y_train)

            # pridction testing data
            y_test_pred=model.predict(X_test)

            # get confusion matrix and precision, recall, f score 
            test_model_score=accuracy_score(y_test,y_test_pred)
            

            report[list(models.keys())[i]]=test_model_score
            report_confusion_matrix[list(models.keys())[i]]=confusion_matrix(y_test,y_test_pred)

            return report, report_confusion_matrix
        
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException (e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)