"""
Author : malzantot (malzantot@ucla.edu)
"""
import _init_paths

import argparse
import tensorflow as tf
import os, sys, cv2
import numpy as np

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')



class DaisDetector(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.net = get_network('VGGnet_test')
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        self.saver.restore(self.sess, self.model_path)

        # Warmup on a dummy image
        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        for i in range(2):
        	_, _= im_detect(self.sess, self.net, im)

    def detect(self, im_file):
        im = cv2.imread(im_file)
        scores, boxes = im_detect(self.sess, self.net, im)
        print ('Detection . # {:d} object proposals'.format(boxes.shape[0]))
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        result = dict()
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
            	pass
            class_result = []
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                class_result.append((bbox, score))
            result[CLASSES[cls_ind]] = class_result
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    dais_detctor = DaisDetector(args.model)

    print("initialzied")
    image_name = '004545.jpg'
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    result = dais_detctor.detect(im_file)
    print(result)
