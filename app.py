from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
from src.components.model_loader import ModelLoader
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def make_predictions():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        model = ModelLoader.loader()
        try:
            '''
            data_1 = float(request.form.get('CIC0')),
            data_2 = float(request.form.get('SM1_Dz(Z)')),
            data_3 = float(request.form.get('GATS1i')),
            data_4 = float(request.form.get('NdsCH')),
            data_5 = float(request.form.get('NdssC')),
            data_6 = float(request.form.get('MLOGP'))
            data = [data_1, data_2, data_3, data_4, data_5, data_6]
            data = np.array(data)
            '''
            data = [float(x) for x in request.form.values()]
            data = np.array(data)
        except:
            logging.info("[app.py] Error Occured while getting data from FORM-HTML[home.html]")
        pred = model.predict(data)
        return render_template('home.html', result = pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
