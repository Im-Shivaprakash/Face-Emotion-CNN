import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend as K

class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation='relu', weightsPath=None):
        model = Sequential()
        
        # Set the input shape based on the data format
        inputShape = (imgRows, imgCols, numChannels)
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        
        # First set of layers: Conv -> Activation -> Pooling
        model.add(Conv2D(6, (5, 5), padding="valid", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Second set of layers: Conv -> Activation -> Pooling
        model.add(Conv2D(16, (5, 5), padding="valid"))
        model.add(Activation(activation))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(1,1)))
        
        # Flattening and Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Activation(activation))
        
        # Output Layer with softmax activation for classification
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        
        # Load weights if a weights path is provided
        if weightsPath is not None:
            model.load_weights(weightsPath)
        
        return model
