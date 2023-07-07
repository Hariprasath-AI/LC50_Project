# The Packages/Methods which are necessary for Model Loading phase are imported here.
from src.components.model_evaluation import ModelEvaluation
from src.utils import Utility
from src.logger import logging

# All operations related to Model Loading phase are carried out inside the class 'ModelLoader'
class ModelLoader:

    '''
    The 'loader' function first tries to load the model in particular loaction. If not available in that location, 
    the evaluate method of class ModelEvaluation is called here.
    '''
    def loader():
        try:
            model = Utility.load('./data/best_model/best_model.pkl')
            logging.info("[model_loader.py] Model is already there. So, loaded from default location")
        except:
            try:
                model = Utility.load('./data/model/model.pkl')
            except:
                ModelEvaluation.evaluate()
                ModelLoader.loader()
            logging.info("[model_loader.py] Model is not present is the default location. So, we're going to develop a model")
        return model