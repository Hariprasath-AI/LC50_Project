# DataTrans
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_loader import ModelLoader
# CustomException is imported from exceptions.py which is available inside 'src' folder 
from src.exceptions import CustomException
from src.logger import logging
from src.utils import Utility
import os
import sys


try:
    _,_,x_test,_ = Utility.import_splitted_data()
    model = ModelLoader.loader()
    pred = model.predict(x_test)
    print(pred)
except Exception as e:
    logging.info("Error occured at somewhere. So, please check !!!")
    CustomException(e, sys)