# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub
import sys
import numpy as np
# For measuring the inference time.
import time
import cv2
import os.path as p
import boxe_drawer as f
# Print Tensorflow version
print(tf.__version__)
# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())



#COMAND ARGUMENTS
INPUT_FILE = ".\MISS DIOR – The new Eau de Parfum.mp4"
OUTPUT_FILE = p.join(p.abspath(p.dirname(INPUT_FILE)),"Detection_"+p.basename(INPUT_FILE))
OUTPUT_FPS = 30
FAST_DETECTION = True
ONLY_HUMAN_DETECTED = True



arguments = sys.argv[1:]
#INPUT FILE
if (len(arguments)>=1):
    INPUT_FILE = arguments[0]
    OUTPUT_FILE = p.join(p.abspath(p.dirname(INPUT_FILE)),"Detection_"+p.basename(INPUT_FILE))

#OUTPUT FILE
if (len(arguments)>=2):
    OUTPUT_FILE = arguments[1]

#OUTPUT FPS
if (len(arguments)>=3):
    OUTPUT_FPS = int(arguments[2])

#FAST DETECTION
if (len(arguments)>=4):
    FAST_DETECTION = (arguments[3]=="true")

#ONLY HUMAN DETECTED
if (len(arguments)>=5):
    ONLY_HUMAN_DETECTED = (arguments[4]=="true")


#Usable detectors
url_inception = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #more acurate 
url_mobilenet = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #faster

if(FAST_DETECTION):
    module_handle = url_mobilenet
else:
    module_handle = url_inception
detector = hub.load(module_handle).signatures['default'] #load the model from tensorflow hub

#The algorithme detects many classes, those are the ones that corespond to detection persons 
human_classes = [b'Woman',b'Girl',b'Man',b'Boy',b'Person']


def run_detector(detector, img, ONLY_HUMAN_DETECTED):
    """run the detection and returns the image with detection boxes """
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img) #Deep Learning Part
    end_time = time.time()
    result = {key:value.numpy() for key,value in result.items()}

    if ONLY_HUMAN_DETECTED:
        image_with_boxes = f.draw_boxes(
            img, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"],white_list = human_classes)
    else:
        image_with_boxes = f.draw_boxes(
            img, result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"])
    return image_with_boxes



#runs the detection for each frame and creates the output video
first = True
FPS = OUTPUT_FPS
out = 0
cap = cv2.VideoCapture(INPUT_FILE)
frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        print("frame :",frame_number)
        if (frame_number == 0):
            #On the first frame setup the output video
            print('Original Video Dimensions : ',frame.shape)    
            width = int(frame.shape[1])
            height = int(frame.shape[0])
            dim = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(OUTPUT_FILE,fourcc, FPS, dim)
            first = False

        img = run_detector(detector, frame,ONLY_HUMAN_DETECTED) #run the detection and draw the frames 

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST )
        out.write(resized)
        frame_number +=1
    else:
        print("noframe")

    cv2.imshow('frame',resized)
    #to stop the process presss "Q"
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
