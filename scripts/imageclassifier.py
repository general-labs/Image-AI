import numpy as np
import urllib.request

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras import backend as K

def classify_image(path):
    with urllib.request.urlopen(path) as url:
        with open('temp/temp.jpg', 'wb') as f:
            f.write(url.read())
    K.clear_session()
    classifier=ResNet50()
    #print(classifier.summary())
    new_image = image.load_img('temp/temp.jpg', target_size=(224, 224))
    transformed_image= image.img_to_array(new_image)
    #print(transformed_image.shape)
    transformed_image=np.expand_dims(transformed_image,axis=0)
    #print(transformed_image.shape)
    transformed_image=preprocess_input(transformed_image)
    #print(transformed_image)
    y_pred= classifier.predict(transformed_image)
    #print(y_pred)
    #print(y_pred.shape)

    decode_predictions(y_pred, top=5)
    label = decode_predictions(y_pred)
    # retrieve the most likely result, i.e. highest probability
    decoded_label = label[0][0]

    print("######===============########")
    # print the classification
    print('%s (%.2f%%)' % (decoded_label[1], decoded_label[2]*100 ))
    print("######===============########")

    # Destroy references
    del classifier, new_image, transformed_image, y_pred, label
    K.clear_session()
    return ({"Prediction": decoded_label[1], "confidence": decoded_label[2] * 100, "url": path})
  