import json, argparse, time

import tensorflow as tf

from flask import Flask, request
from flask_cors import CORS

import matplotlib

from dais_detector import DaisDetector
import os

import numpy as np
import urllib
import cv2

import base64
from PIL import Image
from StringIO import StringIO

app = Flask(__name__)
cors = CORS(app)

def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def URLtoImage(url):
    response = urllib.urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    return image


def DetectionsInImage(image):
    result = dais_detctor.DetectInImage(image)

    return {'cars': [{"pixel_coords":car_result[0].tolist(),"confidence":float(car_result[1])} for car_result in result["car"]]}
    

def DetectInVideo(video,frame_interval):
    video_data = []
    
    more_frames = True
            
    frame_count = -1
    while(1):
        ret, frame = video.read()
        frame_count += 1
        if not frame is None:
            if frame_count == 0 or (frame_count +1) % frame_interval == 0:
                video_data.append({frame_count:DetectionsInImage(frame)})
        else:
            break


    return video_data


@app.route("/tfl-api/detect-cars-video/<string:camera_id>", methods=['GET'])
def GetCarsFromVideoAPI(camera_id):
    api_url = "http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.{}.mp4"
    request_url = api_url.format(str(camera_id))
    video = cv2.VideoCapture(request_url)

    frame_interval = 50

    cars_in_frames = DetectInVideo(video,frame_interval)

    json_data = json.dumps(cars_in_frames)

    return json_data



@app.route("/tfl-api/detect-cars/<string:camera_id>", methods=['GET'])
def GetCarsFromAPI(camera_id):
    #img_path = os.path.join(".","04254","06_1489042033166_frame60.jpg")
    # img_path = os.path.join(".","04503","04503.jpg")
    # result = dais_detctor.detect(img_path)

    api_url = "http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.{}.jpg"
    request_url = api_url.format(str(camera_id))

    img = URLtoImage(request_url)

    show_image = False
    if(show_image):
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    result = dais_detctor.DetectInImage(img)

    json_data = json.dumps({'cars': [{"pixel_coords":car_result[0].tolist(),"confidence":float(car_result[1])} for car_result in result["car"]]})
    
    return json_data


@app.route("/car-detector/image", methods=['POST', 'GET'])
def GetRatingFromImage():
    if request.method == 'POST':
        if 'image' in request.files:
            input_image = cv2.imdecode(numpy.fromstring(request.files['image'].read(), numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)

            result = dais_detctor.DetectInImage(input_image)

            json_data = json.dumps({'cars': [{"pixel_coords":car_result[0].tolist(),"confidence":float(car_result[1])} for car_result in result["car"]]})
            
            return json_data

        if 'image' in request.form.keys():
            input_image = readb64(request.form["image"])

            
            result = dais_detctor.DetectInImage(input_image)

            json_data = json.dumps({'cars': [{"pixel_coords":car_result[0].tolist(),"confidence":float(car_result[1])} for car_result in result["car"]]})
            
            return json_data


        return 'error'
    return '''
    <!doctype html>
    <title>Upload Image File to Detect Cars</title>
    <h1>Upload Image File to Detect Cars</h1>
    <form method=post enctype=multipart/form-data>
    <p><input type=file name=image>
    <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    print("matplotlib check:")
    print(matplotlib.matplotlib_fname())
    print('Loading the model')
    model_path = os.path.join("..","rcnn-models","VGGnet_fast_rcnn_iter_70000.ckpt")
    dais_detctor = DaisDetector(model_path)
    print('Starting the API')
    app.run(host='0.0.0.0', port=5010)