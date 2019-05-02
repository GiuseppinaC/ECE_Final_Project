# VGG_modified
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from keras import backend as K

class vgg_mod:
    @staticmethod
    def build(width, height, depth, classes, finalAct="sigmoid"):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        input = Input(shape=inputShape,name = 'image_input')
        output_vgg16_conv = model_vgg16_conv(input)
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x=Dense(classes, activation='sigmoid')(x)
        
        my_model = Model(inputs=input, outputs=x)
        return my_model

