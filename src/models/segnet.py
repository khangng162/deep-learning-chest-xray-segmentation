import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model

def segnet_model(input_size):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Decoder
    up5 = UpSampling2D((2, 2))(pool4)
    conv5 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = UpSampling2D((2, 2))(conv5)
    conv6 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = UpSampling2D((2, 2))(conv6)
    conv7 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    up8 = UpSampling2D((2, 2))(conv7)
    conv8 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv8)

    model = Model(inputs, outputs)
    return model
