import json, argparse, time

from flask import Flask, request
from flask_cors import CORS

import os

import numpy as np
import urllib
import cv2

import base64
from PIL import Image
from StringIO import StringIO


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
				video_data.append((frame_count,DetectionsInImage(frame)))
		else:
			break


	return video_data


app = Flask(__name__)
cors = CORS(app)
@app.route("/tfl-api/detect-cars-video/<string:camera_id>", methods=['GET'])
def GetCarsFromAPI(camera_id):
	api_url = "http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.{}.mp4"
	request_url = api_url.format(str(camera_id))
	video = cv2.VideoCapture(request_url)

	frame_interval = 10

	cars_in_frames = DetectInVideo(video,frame_interval)

	json_data = json.dumps(cars_in_frames)

	return json_data


if __name__ == "__main__":
    print('Starting the API')
    app.run(host='0.0.0.0', port=5060)