from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
from src.components.model_loader import ModelLoader
from src.logger import logging

application=Flask(__name__)

@application.route('/')
def home():
    return render_template('home.html')

@application.route('/predict', methods=['GET', 'POST'])
def make_predictions():
    if request.method=='GET':
        return render_template('home.html')
    elif request.method=='POST':
        model = ModelLoader.loader()
        try:
            data = [float(x) for x in request.form.values()]
            data = np.array(data)
        except:
            logging.info("[app.py] Error Occured while getting data from FORM-HTML[home.html]")
        pred = model.predict(data)
        return render_template('home.html', preds=pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
