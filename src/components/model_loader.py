from src.components.model_evaluation import ModelEvaluation
from src.utils import Utility
from src.logger import logging

class ModelLoader:
    def loader():
        try:
            model = Utility.load('./data/model/best_model.pkl')
            logging.info("[model_loader.py] Model is already there. So, loaded from default location")
        except:
            ModelEvaluation.evaluate()
            logging.info("[model_loader.py] Model is not present is the default location. So, we're going to develop a model")
        return model