# DataTrans
from src.components.data_transformation import DataTransformation
# CustomException is imported from exceptions.py which is available inside 'src' folder 
from src.exceptions import CustomException
import os
import sys

try:
    print(DataTransformation.train_test_splitting())
except Exception as e:
    CustomException(e, sys)