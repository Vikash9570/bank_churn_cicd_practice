import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.pipeline.training_pipeline import calling_function

calling_function()