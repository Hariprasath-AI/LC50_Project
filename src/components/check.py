# DataTrans
from src.components.data_transformation import DataTransformation
# CustomException is imported from exceptions.py which is available inside 'src' folder 
from src.exceptions import CustomException
import os
import sys

try:
    rec = DataTransformation.handling_missing_values()
    print(rec)
except Exception as e:
    CustomException(e, sys)