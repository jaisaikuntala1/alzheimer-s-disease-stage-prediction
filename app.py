from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
img_height, img_width=224,224

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



def predictImage(file_path):
    names = file_path.split('\\')
    testimage = names[len(names)-1]
    testimage = 'uploads/'+testimage
    print("TEST IMAGE:"+testimage)
    img = image.load_img(testimage, target_size=(img_height, img_width))
    new_model = load_model('models/final_model.h5')
    #MobileNetModelImagenet.h5
    img = img_to_array(img)
    samples_to_predict = []
    samples_to_predict.append(img)
    samples_to_predict = np.array(samples_to_predict)
    print(samples_to_predict.shape)
    predictions = new_model.predict(samples_to_predict)
    class_to_label = {
        0:'MildDemented',
        1:'ModerateDemented',
        2:'NonDemented',
        3:'VeryMildDemented'
    }
    index = 0
    maxSoFar = predictions[0][0]
    for i in range(1,4):
        if maxSoFar < predictions[0][i]:
            maxSoFar = predictions[0][i]
            index = i
    print("prediction : ", class_to_label[index])
    print("TEST IMAGE:", testimage)
    print("shape : ", img.shape)
    return class_to_label[index]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = predictImage(file_path)
        print("Image class : ", preds)
        return "The given brain MRI image is "+preds
    return None


if __name__ == '__main__':
    app.run(debug=True)