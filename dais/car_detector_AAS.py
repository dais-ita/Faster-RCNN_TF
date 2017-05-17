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



def URLtoImage(url):
    response = urllib.urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    return image


app = Flask(__name__)
cors = CORS(app)
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

if __name__ == "__main__":
    print("matplotlib check:")
    print(matplotlib.matplotlib_fname())
    print('Loading the model')
    model_path = os.path.join("..","rcnn-models","VGGnet_fast_rcnn_iter_70000.ckpt")
    dais_detctor = DaisDetector(model_path)
    print('Starting the API')
    app.run(host='0.0.0.0')