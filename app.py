from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#newly added
import numpy as np
import pandas as pd
from PIL import  Image

# from data.create_data import create_table

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# st.set_option('deprecation.showfileUploaderEncoding', False)

app = Flask(__name__)

#WEIGHTS = '/home/fiifi/Desktop/Lab/AI_Lab/AIDataset/brain-mri-dataset/brain_tumor_classifier/brain_tumor_predictor/models/classifier.h5'
cwd = os.getcwd()

# model_weight = cwd + '/models/20211127-02161637979419-greatXrayCTMultiClassCovid19Model.h5'
# model_weight2 = cwd + '/models/20211113-21011636837298-Covid19-XRayDetection-Model-Good-2 (1).h5'
# model_weight3 = cwd + '/models/greatCTCovid19ModelGC.h5'

#print(model_weight)

# model = load_model(model_weight)
# model2 = load_model(model_weight2)
# model3 = load_model(model_weight3)

print('Model loaded. Check http://127.0.0.1:5000/')

# Predict Function


def model_prediction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    image_data = np.expand_dims(img_array, axis=0)
    image_data = preprocess_input(image_data)
    pred = model.predict(image_data)

    return pred


def model_prediction2(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    image_data = np.expand_dims(img_array, axis=0)
    image_data = preprocess_input(image_data)
    pred = model.predict(image_data)

    return pred


def model_prediction3(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    image_data = np.expand_dims(img_array, axis=0)
    image_data = preprocess_input(image_data)
    pred = model.predict(image_data)

    return pred


# newly added
# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224, 224), Image.ANTIALIAS)
    # image = ImageOps.fit(image_data, (224, 244), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    # st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

    # model = tf.keras.models.load_model('my_model2.h5')
# modelst = tf.keras.models.load_model('models/20211127-02161637979419-greatXrayCTMultiClassCovid19Model.h5')
# modelstxray = tf.keras.models.load_model('models/20211113-21011636837298-Covid19-XRayDetection-Model-Good-2 (1).h5')
# modelstct = tf.keras.models.load_model('models/greatCTCovid19ModelGC.h5')

modelst = tf.keras.models.load_model('greatXrayCTMultiClassCovid19Model2')
modelstxray = tf.keras.models.load_model('Covid19detectionXraymodelgood22')
modelstct = tf.keras.models.load_model('greatCTCovid19ModelGC2')


@app.route("/", methods=['GET'])
def index():
    return render_template('index2.html')

@app.route("/xray", methods=['GET'])
def indexseeko():
    return render_template('xrays.html')

@app.route("/ct", methods=['GET'])
def indexbeeko():
    return render_template('cts.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        img_file = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(img_file.filename))
        img_file.save(file_path)

        # Prediction
        # result = model_prediction(file_path, model)

        #newly added
        imageIM = Image.open(img_file)
        # imageIM = Image.open(file_path)
        # st.image(imageIM, use_column_width=True)
        # st.write(file_details)
        # st.image(load_image(image_file), width=250)
        prediction = import_and_predict(imageIM, modelst)
        # pred = prediction[0][0]
        pred = prediction[0][0]
        if pred == np.max(prediction):
            # if (pred > 0.5):
            # st.write("""
            #                              ## **Prediction:** Covid19 Detected in CT!
            #                              """
            #          )
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            print(prediction)
            # print("the prediction result is: ", result)
            #     print("pred is: ", pred)
            return 'Covid19 Detected! in CT'
        elif prediction[0][1] == np.max(prediction):
            # st.write("""
            #                              ## **Prediction:** Normal and healthy chest CT
            #                              """
            #          )
            # st.balloons()
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            print(prediction)
            # print("the prediction result is: ", result)
            #     print("pred is: ", pred)
            return 'Normal and healthy chest CT 👍'
        elif prediction[0][2] == np.max(prediction):
            # st.write("""
            #                                              ## **Prediction:** Covid19 Detected in Xray!
            #                                              """
            #          )
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            print(prediction)
            # print("the prediction result is: ", result)
            #     print("pred is: ", pred)
            return 'Covid19 Detected! in Xray'
        elif prediction[0][3] == np.max(prediction):
            # st.write("""
            #                                              ## **Prediction:** Normal and healthy chest Xray
            #                                              """
            #          )
            # st.balloons()
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            print(prediction)
            #     print("the prediction result is: ", result)
            #     print("pred is: ", pred)
            return 'Normal and healthy chest Xray 👍'

        # if result[0][0] == 1:
        #     return 'Covid19 Detected!'
        # else:
        #     return 'Normal and healthy chest'
        # pred = result[0][0]
        # # if result == 0:
        # #     return 'Covid19 Detected!'
        # # else:
        # #     return 'Normal and healthy chest 👍'
        #
        # if pred == np.max(result):
        # # # if (pred > 0.5):
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Covid19 Detected! in CT'
        # elif result[0][1] == np.max(result):
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Normal and healthy chest CT 👍'
        # elif result[0][2] == np.max(result):
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Covid19 Detected! in Xray'
        # elif result[0][3] == np.max(result):
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Normal and healthy chest Xray 👍'

    return None



@app.route('/predict2', methods=['GET','POST'])
def uploadseeko():
    if request.method == 'POST':
        img_file = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(img_file.filename))
        img_file.save(file_path)

        #newly added
        imageIM = Image.open(img_file)
        # imageIM = Image.open(file_path)
        # st.image(imageIM, use_column_width=True)
        # st.write(file_details)
        # st.image(load_image(image_file), width=250)
        prediction = import_and_predict(imageIM, modelstxray)
        pred = prediction[0][0]
        print(prediction)
        print("pred only is: ", pred)
        if prediction < 0.5:
            # if (pred > 0.5):
            # st.write("""
            #                                  ## **Prediction:** Covid19 Detected!
            #                                  """
            #          )
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            return 'Covid19 Detected!'
        else:
            # st.write("""
            #                                  ## **Prediction:** Normal and healthy chest
            #                                  """
            #          )
            # st.balloons()
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            return 'Normal and healthy chest 👍'

        # Prediction
        # result = model_prediction2(file_path, model2)

        # if result[0][0] == 1:
        #     return 'Covid19 Detected!'
        # else:
        #     return 'Normal and healthy chest'

        # if result == 0:
        #     print("the prediction result is: ", result)
        #     return 'Covid19 Detected! seeko'
        # else:
        #     print("the prediction result is: ", result)
        #     return 'Normal and healthy chest seeko 👍'

    return None


@app.route('/predict3', methods=['GET','POST'])
def uploadbeeko():
    if request.method == 'POST':
        img_file = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(img_file.filename))
        img_file.save(file_path)

        #newly added
        imageIM = Image.open(img_file)
        # imageIM = Image.open(file_path)
        # st.image(imageIM, use_column_width=True)
        # st.write(file_details)
        # st.image(load_image(image_file), width=250)
        prediction = import_and_predict(imageIM, modelstct)
        pred = prediction[0][0]
        print(prediction)
        print("pred only is: ", pred)
        # maybe change below to < 0.5 instead
        if pred == np.max(prediction):
            # if (pred > 0.5):
            # st.write("""
            #                              ## **Prediction:** Covid19 Detected!
            #                              """
            #          )
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            return 'Covid19 Detected!'
        else:
            # st.write("""
            #                              ## **Prediction:** Normal and healthy chest
            #                              """
            #          )
            # st.balloons()
            # new_space = '<br><br><hr>'
            # st.markdown(new_space, unsafe_allow_html=True)
            return 'Normal and healthy chest 👍'

        # Prediction
        # result = model_prediction3(file_path, model3)
        # pred = result[0][0]
        # if result[0][0] == 1:
        #     return 'Covid19 Detected!'
        # else:
        #     return 'Normal and healthy chest'
        # if result == 0:

        # if pred == np.max(result):
        #     # if (pred > 0.5):
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Covid19 Detected! beeko'
        # else:
        #     print("the prediction result is: ", result)
        #     print("pred is: ", pred)
        #     return 'Normal and healthy chest beeko 👍'

    return None



if __name__ == "__main__":
    app.run()