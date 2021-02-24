# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
import PIL
import pandas as pd

sys.path.append("..")

# Import utilites
from field_detection.utils import label_map_util
from field_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'field_detection/inference_graph'
#IMAGE_NAME = 'aug1_62.jpeg'

# current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'field_detection/training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 5

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#label_map = label_map_util.load_labelmap("training/labelmap.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors for the object detection classifier
def getFields(image):
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    image_expanded = np.expand_dims(image, axis=0)

   
    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})
    print(boxes[0])
    print(scores[0])
    print(classes[0])
    print(num[0])

    new_image=image.copy()
# Draw the results of the detection

    vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=3,
    min_score_thresh=0.50)

    #print(image)
    
    #preprocessed_image=cv2.imread("invert_final_T4.jpg")
    #print(new_image)
    height, width, channel = image.shape
    #print(height)
    #print(width)
    #print(channel)
    field_list=[]
    for (box, score, class_) in zip(boxes[0], scores[0], classes[0]):
        if score>0.50:
            field_dict={}
            ymin = (int(box[0]*height))
            xmin = (int(box[1]*width))
            ymax = (int(box[2]*height))
            xmax = (int(box[3]*width))

            croppedImage = new_image[ymin:ymax,xmin:xmax]
            
            label = category_index[class_]['name']
            #field_dict[label]=croppedImage
            field_dict["class_name"]=label
            field_dict["value"]=croppedImage
            field_list.append(field_dict)
            #print(label)
            print(field_list)

    return field_list

#image = cv2.imread("1.jpg")
#getFields(image)
