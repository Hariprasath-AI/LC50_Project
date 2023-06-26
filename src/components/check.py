# DataTrans
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
# CustomException is imported from exceptions.py which is available inside 'src' folder 
from src.exceptions import CustomException
from src.logger import logging
import os
import sys

try:
    ModelTrainer.generate_report()
except Exception as e:
    logging.info("Error occured at somewhere. So, please check !!!")
    CustomException(e, sys)