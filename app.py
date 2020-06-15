import model
from flask import Flask, request, redirect, url_for, render_template, Response
from flask_bootstrap import Bootstrap

import zipfile
import plotly
import json
import os
import numpy as np
import urllib.request
import glob

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

app = Flask(__name__)
bootstrap = Bootstrap(app)


OUTPUT_DIR = 'tmp'

import shutil

def empty_folder(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    os.rmdir(path)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        if filename != '':
            if uploaded_file.filename[-3:] in ['zip']:

                if not os.path.isdir(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)

                image_path = os.path.join(OUTPUT_DIR, filename)
                uploaded_file.save(image_path)

                with zipfile.ZipFile(image_path, 'r') as zip_ref:
                    zip_ref.extractall(OUTPUT_DIR)

                img_dir = OUTPUT_DIR + "/" + filename[:-4]
                dicom_paths = model.get_dicom_paths(img_dir)

                pet_scan = model.get_pet_scan(dicom_paths)

                prediction = model.get_prediction(pet_scan)
                if prediction[1] > 0.5:
                    diagnosis = 'Alzheimer\'s Disease'
                else:
                    diagnosis = 'Cognitively Normal'

                classidx = np.argmax(prediction)

                heat_map = model.compute_saliency_map(pet_scan, classidx = classidx)

                fig = model.create_plot(pet_scan, heat_map)

                fig_json=json.dumps({'data': json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
                                    'layout': json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder))})


                empty_folder(OUTPUT_DIR)
                return render_template('show.html', prediction = np.round(prediction[classidx], decimals=2), diagnosis = diagnosis, plot_json = fig_json)
        #else:

            #url = 'https://drive.google.com/uc?export=download&id=1DaL2KOLAZ616MFPOgwCQdqEeoUFi2cRa'
            #urllib.request.urlretrieve(url, 'file.zip')
            #with zipfile.ZipFile('file.zip', 'r') as zip_ref:
            #    zip_ref.extractall(OUTPUT_DIR)
            #os.remove('file.zip')

            ##img_dir = glob.glob(OUTPUT_DIR + '/*')[1]
            #img_dir = OUTPUT_DIR + '/I257156'

            #dicom_paths = model.get_dicom_paths(img_dir)

            #pet_scan = model.get_pet_scan(dicom_paths)

            #prediction = model.get_prediction(pet_scan)
            #if prediction[1] > 0.5:
            #    diagnosis = 'Alzheimer\'s Disease'
            #else:
            #    diagnosis = 'Cognitively Normal'

            #classidx = np.argmax(prediction)

            #heat_map = model.compute_saliency_map(pet_scan, classidx = classidx)

            #fig = model.create_plot(pet_scan, heat_map)
            #redata = json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder))
            #relayout = json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder))

            #fig_json=json.dumps({'data': redata,'layout': relayout})

            #empty_folder(OUTPUT_DIR)

            #return render_template('show.html', prediction = 0.84, diagnosis = 'Alzheimer\'s Disease', plot_json = fig_json)


    return render_template('index.html')

@app.route('/AD_example')
def show_ad_example():
    with open('static/ad_fig.txt', 'r') as myfile:
        fig_json = myfile.read()
    return render_template('show.html', prediction = 0.84, diagnosis = 'Alzheimer\'s Disease', plot_json = fig_json)

@app.route('/CN_example')
def show_cn_example():
    with open('static/cn_fig.txt', 'r') as myfile:
        fig_json = myfile.read()
    return render_template('show.html', prediction = 0.87, diagnosis = 'Cognitively Normal', plot_json = fig_json)

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

if __name__ == '__main__':
    app.run()
