import json, argparse, time

import tensorflow as tf

from flask import Flask, request
from flask_cors import CORS

from dais_detector import DaisDetector
import os

app = Flask(__name__)
cors = CORS(app)
@app.route("/api/predict", methods=['GET'])
def predict():
    start = time.time()
    print(start)
    #add get request processing for camera_id
    img_path = os.path.join(".","04254","06_1489042033166_frame60.jpg")

    result = dais_detctor.detect(img_path)

    json_data = json.dumps({'cars': [{"pixel_coords":car_result[0].tolist(),"confidence":float(car_result[1])} for car_result in result["car"]]})
    print("Time spent handling the request: %f" % (time.time() - start))
    
    return json_data

if __name__ == "__main__":
    print('Loading the model')
    model_path = os.path.join("..","rcnn-models","VGGnet_fast_rcnn_iter_70000.ckpt")
    dais_detctor = DaisDetector(model_path)
    print('Starting the API')
    app.run()