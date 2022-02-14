from __future__ import division, print_function
import shutil
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


if os.environ.get('app_env') == "production":
    from google.cloud import storage
    storage_client= storage.Client()
    bucket_name = 'fgg-models'
    bucket = storage_client.bucket(bucket_name=bucket_name)
    
from pathlib import Path

#newly added
import numpy as np
# import pandas as pd
from PIL import  Image

# from data.create_data import create_table

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# st.set_option('deprecation.showfileUploaderEncoding', False)

application = app = Flask(__name__)

#WEIGHTS = '/home/fiifi/Desktop/Lab/AI_Lab/AIDataset/brain-mri-dataset/brain_tumor_classifier/brain_tumor_predictor/models/classifier.h5'
cwd = os.getcwd()

# model_weight = cwd + '/20211127-02161637979419-greatXrayCTMultiClassCovid19Model.h5'
# model_weight2 = cwd + '/20211113-21011636837298-Covid19-XRayDetection-Model-Good-2 (1).h5'
# model_weight3 = cwd + '/greatCTCovid19ModelGC.h5'

#print(model_weight)

# model = load_model(model_weight)
# model2 = load_model(model_weight2)
# model3 = load_model(model_weight3)

# print('Model loaded. Check http://127.0.0.1:5000/')

# Predict Function




def download_folder(prefix):
    """Downloads the folder from the bucket"""
    print("[INFO] Downloading folder from bucket")
    if os.path.exists(prefix): return
    print("[INFO] Really Downloading")
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name) 

if os.environ.get('app_env') == "production":
    if os.path.exists("greatXrayCTMultiClassCovid19Model2"): shutil.rmtree("greatXrayCTMultiClassCovid19Model2")

    download_folder("greatXrayCTMultiClassCovid19Model2")





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
# modelst = tf.keras.models.load_model('20211127-02161637979419-greatXrayCTMultiClassCovid19Model.h5')
# modelstxray = tf.keras.models.load_model('20211113-21011636837298-Covid19-XRayDetection-Model-Good-2 (1).h5')
# modelstct = tf.keras.models.load_model('greatCTCovid19ModelGC.h5')

print("[INFO] - Loading Models")



modelst = tf.keras.models.load_model('greatXrayCTMultiClassCovid19Model2')
modelstxray = tf.keras.models.load_model('greatXrayCTMultiClassCovid19Model2')
modelstct = tf.keras.models.load_model('greatXrayCTMultiClassCovid19Model2')

print("[INFO] - Model loaded")

@application.route("/")
def index():
    return render_template('index2.html')

@application.route("/xray")
def indexseeko():
    return render_template('xrays.html')

@application.route("/ct")
def indexbeeko():
    return render_template('cts.html')


@application.route('/predict', methods=['POST'])
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



@application.route('/predict2', methods=['POST'])
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
        if pred < 0.5:
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


@application.route('/predict3', methods=['POST'])
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
    app.run(debug=True)