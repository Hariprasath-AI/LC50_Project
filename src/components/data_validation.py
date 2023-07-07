# The Packages/Methods which are necessary for Data validation phase are imported here.
import os
import sys  
import pandas as pd 
from src.logger import logging 
from src.exceptions import CustomException 

'''
A class named 'DataValidation' contains the method/function 'validate' to check whether the data is in .csv format or not.
If it is not in .csv format, then the data is not imported and throws an exception and logged.
The Project got struck here.
'''
class DataValidation:
    def validate():
        try:
            loc = 'data\dataset(csv)\qsar_fish_toxicity.csv'
            column_names = ['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP', 'LC50']
            data = pd.read_csv(loc, header = None, delimiter = ';', names = column_names)
            logging.info("[data_validation.py] There is no problem with the data. So, we can continue further. The data passed 'validate()'")
            logging.info("Data Validation is completed successfully")
            return data
        except Exception as e:
            logging.info("[data_validation.py] Error occured while importing the data. Please check format of the data i.e., csv")
            CustomException(e,sys)
        logging.info("Data Validation Operations Ends here.....")
        
    
            