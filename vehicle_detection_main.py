#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from tkinter.filedialog import *

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Movement Direction, Vehicle Speed (km/h), Time (s)'
    writer.writerows([csv_line.split(',')])

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )


# input video or use webcam

if sys.argv[1] == "use_cam" :
    cap = cv2.VideoCapture(1) #0 for the built-in webcam 1 for another one plug in USB
if  sys.argv[1] == "use_videofile" :
    filepath = askopenfilename(title="Open your video",filetypes=[('mkv files','.mkv'), ('mp4 files','.mp4'), ('avi files','.avi'),('all files','.*')])
    cap = cv2.VideoCapture(filepath)




width = cap.get(3)
height = cap.get(4)

ROI_height = height/1.7
#print(int(ROI_height))

#Time of the detected object
time=  cap.get(0)

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'   #'ssdlite_mobilenet_v2_coco_2018_05_097'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt') 

#oid_bbox_trainable_label_map.pbtxt
#pascal_label_map.pbtxt

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            1)).astype(np.uint8)


# Detection
def object_detection_function():
    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    time_s = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                time_ms = cap.get(0)
                time_s = int(time_ms/1000)
                time = str(time_s)
                

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                #print([category_index.get(i) for i in classes[0]])
                #print(scores)

                # Visualization of the results of a detection.
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    int(ROI_height),
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    time_s,
                    use_normalized_coordinates=True,
                    line_thickness=3)

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                line_width = int(width);
                line_height = int(height/1.1);
                if counter == 1:
                    cv2.line(input_frame, (0, int(ROI_height)), (line_width, int(ROI_height)), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, int(ROI_height)), (line_width, int(ROI_height)), (0, 0, 0xFF), 5)
                # insert information text to video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    ((int)(line_width*0.8), int(ROI_height) -30),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,

                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Movement Direction: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Speed(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Vehicle Size/Type: ' + size,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Time: ' + time + ' s',
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                cv2.imshow('vehicle detection', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, direction, speed, time) = csv_line.split(',')

                        writer.writerows([csv_line.split(',')])

                
            cap.release()
            cv2.destroyAllWindows()


object_detection_function()		
