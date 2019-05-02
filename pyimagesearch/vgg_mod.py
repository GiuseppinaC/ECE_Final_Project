# VGG_modified
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

class vgg_mod:
	@staticmethod
	def build(width, height, depth, classes, finalAct="sigmoid"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		
        #model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        #Create  input format 
        input = Input(shape=inputShape,name = 'image_input')
        

        #Use the generated model 
        output_vgg16_conv = model_vgg16_conv(input)
        #Add the fully-connected layers 
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        

		# sigmoid classifier
		x.add(Dense(classes))
		x.add(Activation(finalAct))

		# return the constructed network architecture
		return x
