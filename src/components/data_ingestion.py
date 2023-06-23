# DataValidation class is imported from src.components.data_validation to validate the incoming data
from src.components.data_validation import DataValidation
# logging method is imported from src.logger
from src.logger import logging

class DataIngestion:
    def data_ingestion():
        # This is the constant, where 'DATA_LOC' holds the location of the data.
        DATA_LOC = 'data\dataset(csv)\qsar_fish_toxicity.csv'

        # If the data is valid then it will be returned from vadidate method of src.components.data_validation.DataValidation
        # Otherwise the error wil be logged in the logs folder
        data = DataValidation.validate(DATA_LOC)

        # The validated data is returned to the further component.
        return data

# Have to log that everything is fine is the data ingestion.
logging.info("Everything will working fine until data ingestion phase")