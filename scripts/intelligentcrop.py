import numpy as np
import copy, cv2, getopt, math, sys
from matplotlib import pyplot as plt
import imutils
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import os
from scipy import misc
from imageai.Detection import ObjectDetection
import align.detect_face
import urllib.request


#docker memory fix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('model/face_detect_model/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False)

def detect_faces(img,gpu_memory_fraction,margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face, cropping may be wrongly done please maunually crop this image")
        return None
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    return bb

def smart_crop():
    
    height = 1000
    width  = 665

    input_image = 'static/temp.jpg'
    output_image = 'static/output' + sys.argv[1] + '.jpg'

    print("the script has the name %s" % (sys.argv[0]))
    print ("the script has the name %s" % (sys.argv[1]))

    print('=====*****======')
    print(input_image)
    print(output_image)
    execution_path = os.getcwd()
    print(execution_path)
    print('=====*****======')

    image = misc.imread(input_image)
    image = imutils.resize(image, height=1000)
    original = copy.copy(image)
    face_cords = detect_faces(image,1.0,44)
    if face_cords is None:
        print("can't detect face, cropping may be wrongly done please maunually crop this image")
    else:
        print("detected a person in the image doing smart cropping of image to given size")

        detections = detector.detectCustomObjectsFromImage(input_type='array',input_image=image,
                                                   output_type='array',
                                                   custom_objects=custom_objects, 
                                                   minimum_percentage_probability=65,
                                                  )
        object_cords = detections[1][0]['box_points']

        maxFaceCenter = int(face_cords[0]+(face_cords[2]-face_cords[0])/2)

        if maxFaceCenter-object_cords[0] > object_cords[2]-maxFaceCenter:
            print("Not aligned move the maxface to right")
            maxFaceCenter = maxFaceCenter+(maxFaceCenter-object_cords[0]) - (object_cords[2]-maxFaceCenter)

        right_dist = image.shape[1] - int(maxFaceCenter)
        left_dist = int(maxFaceCenter)


        if left_dist < int(width/2):
            crop = 'left'
            print("object is to the left of the image cropping appropriately")
            x1,y1,x2,y2 = 0,0,int(maxFaceCenter+(width/2)+(width/2-left_dist)),1000
            if x2-x1 !=665:
                x2 = x2 + (665-(x2-x1))

        elif right_dist < int(width/2):
            crop = 'right'
            print("object is to the right of the image cropping appropriately")
            x1,y1,x2,y2 = (int(maxFaceCenter-(width/2))-int(width/2-right_dist)),0,image.shape[1],1000
            if x2-x1 !=665:
                x1 = x1 - (665-(x2-x1))

        else:
            crop = 'center'
            x1,y1,x2,y2 = int(maxFaceCenter-width/2),0,int(maxFaceCenter+width/2),1000
        croppedData = original[y1:y2, x1:x2]
        print("Output shape after cropping",croppedData.shape[:2])
        misc.imsave(output_image, croppedData)
    return "DONE"

smart_crop()
#smart_crop('input-image/obama.jpg', 'output-image/ouput.jpg')
