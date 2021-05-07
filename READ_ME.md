<T1>HOME EXERCISE ANSWER SCRIPT</T1>

This folder contains the original video, the video with the detection boxes and the script used for the detection.

To recreate the detection video you need to run the person_detection.py script.
By default the scripts uses the mp4 video in this folder but you can input other files either by inputing the argument in command line or by modifying the script.

You can also modify the output file, modify its fps, choose the model used and enable all the classes detected by those models     

It uses the boxe_drawer.py file to decluter the main script from boxe drawing functions.

The models are downloaded by the script from tensorflow hub.

You can notice that I only kept  labels corresponding to Woman, Girl, Man, Boy and Person. The models usualy detect Natalie Portman as both a girl and a woman at the same time. This could be resolved by merging boxes with similar labels and coordonates.


The test environment is
    python 3.7.10
    numpy
    cv2
    Pillow
    tensorflow_hub
     tensorflow-gpu 2.4.0 or tensorflow 2.4.0