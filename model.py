import tensorflow as tf
import io
import numpy as np
import pandas as pd

import os
import string
import random
import json
import requests
import glob
import pydicom
from scipy import ndimage, misc

import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots

def get_model():
    model1 = tf.keras.models.load_model('static/model/3dcnn_model_dense16_98_78_60_batch32_min_loss.h5')
    model2 = tf.keras.models.load_model('static/model/3dcnn_model_gap_98_78_60_batch32_min_loss_epoch600.h5')
    return model1, model2


def get_dicom_paths(img_dir):
    dicom_paths = glob.glob(img_dir+'/*')

    slices = []
    for path in dicom_paths:
        n_slice = path.split('_')[-3]
        left = path.find('_' + n_slice + '_') + 1
        right = left + len(n_slice)
        slices.append([path[:left], int(n_slice), path[right:]])

    # Order slices
    slice_df = pd.DataFrame(slices, columns = ['col1', 'slice', 'col3']).sort_values(by='slice')

    slice_df['combined'] = slice_df['col1'] + slice_df['slice'].map(lambda x: str(x)) + slice_df['col3']

    return slice_df['combined'].values

def get_pet_scan(dicom_paths):
    slices = []
    for path in dicom_paths:
        img = pydicom.read_file(path)
        img_array = img.pixel_array

        img_array = np.expand_dims(img_array, axis = -1)
        img_array = tf.cast(img_array, tf.float32)
        slices.append(img_array)

    # Take 75 slices by pruning the first 11 and last 10 slices.
    pet_scan = np.concatenate(slices[16:76], axis=-1)
    #pet_scan[pet_scan < 500] = 0

    # Scale the images.
    minimum = np.min(pet_scan)
    maximum = np.max(pet_scan)
    pet_scan = (pet_scan - minimum)/(maximum - minimum)

    # Crop out the empty part of the image
    pet_scan = pet_scan[30:128,40:118,:]

    return np.expand_dims(pet_scan, axis = -1)

def get_prediction(pet_scan):
    pet_scan = np.expand_dims(pet_scan, axis=0)

    model1, model2 = get_model()

    prediction = (model1.predict(pet_scan).squeeze() + model2.predict(pet_scan).squeeze())/2
    return prediction


def create_plot(petscan, heatmap, num_slices = 60):

    # Create figure
    fig = make_subplots(rows=1,
                        cols=3,
                        subplot_titles = ('Original PET scan', 'Saliency Map', 'Overlayed'))
    #fig = go.Figure()
    zmax = np.max(heatmap)
    zmin = np.min(heatmap)

    # Add traces, one for each slider step
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=petscan[::-1,:,step,0], colorscale = 'Rainbow', showscale = False),
            row=1, col=1)
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=heatmap[::-1,:,step], colorscale = 'Plasma', zmax = zmax, zmin = zmin),
            row=1, col=2)
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=petscan[::-1,:,step,0], colorscale = 'gray', showscale = False),
            row=1, col=3)
    for step in range(num_slices):
        fig.add_trace(
            go.Heatmap(
                z=heatmap[::-1,:,step], colorscale = 'Plasma', zmax = zmax, zmin = zmin, opacity = 0.8),
            row=1, col=3)

    fig.data[0].visible = True
    fig.data[num_slices].visible = True
    fig.data[2*num_slices].visible = True
    fig.data[3*num_slices].visible = True



    # Create and add slider
    slices = []
    for i in range(num_slices):
        slice = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label='Slice {}'.format(i)
        )
        slice['args'][0]['visible'][i] = True
        slice['args'][0]['visible'][i+num_slices] = True
        slice['args'][0]['visible'][i+2*num_slices] = True
        slice['args'][0]['visible'][i+3*num_slices] = True

        slices.append(slice)



    sliders = [dict(
        active=0,
        #currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=slices
    )]

    fig.update_layout(
        width = 1170,
        height = 490,
        sliders=sliders
    )

    return fig

def scale_array(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

def vanilla_backprop(pet_scan, model, classidx = 1):

    inputs = tf.cast(np.expand_dims(pet_scan, axis = 0), tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        prediction = model(inputs)
        loss = prediction[:,classidx]

    grads = tape.gradient(loss, inputs)[0]

    backprop_heatmap = grads[:,:,:,0].numpy()

    #backprop_heatmap[backprop_heatmap < 0] = 0
    #backprop_heatmap = np.abs(backprop_heatmap)
    positive_heatmap = np.maximum(backprop_heatmap, 0)
    negative_heatmap = np.maximum(-backprop_heatmap, 0)
    backprop_heatmap = np.abs(backprop_heatmap)

    backprop_heatmap = scale_array(backprop_heatmap)
    positive_heatmap = scale_array(positive_heatmap)
    negative_heatmap = scale_array(negative_heatmap)


    return backprop_heatmap, positive_heatmap, negative_heatmap

def compute_saliency_map(pet_scan, classidx = 1):

    model1, model2 = get_model()

    backprop_heatmap1, positive_heatmap1, negative_heatmap1 = vanilla_backprop(pet_scan, model1, classidx = classidx)
    backprop_heatmap2, positive_heatmap2, negative_heatmap2 = vanilla_backprop(pet_scan, model2, classidx = classidx)
    backprop_heatmap = (backprop_heatmap1+backprop_heatmap2)/2
    #negative_heatmap = (negative_heatmap1+ negative_heatmap2)/2
    #positive_heatmap = (positive_heatmap1+ positive_heatmap2)/2

    #pmax = np.max(positive_heatmap)
    #pmin = np.min(positive_heatmap)

    #nmax = np.max(negative_heatmap)
    #nmin = np.min(negative_heatmap)

    return scale_array(backprop_heatmap)
