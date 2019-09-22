import keras
import numpy as np
import urllib.request
import cv2

def classify_image(path):
    from keras.applications import vgg16, inception_v3, resnet50, mobilenet
    vgg_model = vgg16.VGG16(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.imagenet_utils import decode_predictions
    from keras import backend as K

    url_image = 'https://www.princess.com/images/ships-and-experience/ships/ship-snippet-640-v2.jpg'
    with urllib.request.urlopen(path) as url:
        with open('temp/temp.jpg', 'wb') as f:
            f.write(url.read())

    filename = 'temp/temp.jpg'

    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))

    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = vgg_model.predict(processed_image)
    
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label_vgg = decode_predictions(predictions)

    # prepare the image for the ResNet50 model
    processed_image = resnet50.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = resnet_model.predict(processed_image)

    # convert the probabilities to class labels
    # If you want to see the top 3 predictions, specify it using the top argument
    label_resnet = decode_predictions(predictions, top=3)

    # prepare the image for the MobileNet model
    processed_image = mobilenet.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = mobilenet_model.predict(processed_image)

    # convert the probabilities to imagenet class labels
    label_mobilenet = decode_predictions(predictions)

    # load an image in PIL format
    original = load_img(filename, target_size=(299, 299))

    # Convert the PIL image into numpy array
    numpy_image = img_to_array(original)

    # reshape data in terms of batchsize
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the Inception model
    processed_image = inception_v3.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = inception_model.predict(processed_image)

    # convert the probabilities to class labels
    label_inception = decode_predictions(predictions)

    K.clear_session()

    return ({
      "VGG16": [label_vgg[0][0][1], str(int(label_vgg[0][0][2] * 100))],
      "MobileNet": [label_mobilenet[0][0][1], str(int(label_mobilenet[0][0][2] * 100))],
      "Inception": [label_inception[0][0][1], str(int(label_inception[0][0][2] * 100))],
      "ResNet50": [label_resnet[0][0][1], str(int(label_resnet[0][0][2] * 100))],
      })


