import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout

def unetpp_model(input_size):
    inputs = Input(input_size)

    # Encoder
    conv1_0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1_0 = BatchNormalization()(conv1_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    conv2_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2_0 = BatchNormalization()(conv2_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    conv3_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3_0 = BatchNormalization()(conv3_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    conv4_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_0 = BatchNormalization()(conv4_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_0)

    conv5_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5_0 = BatchNormalization()(conv5_0)

    # Decoder with nested skip connections (U-Net++ structure)
    up4_1 = UpSampling2D(size=(2, 2))(conv5_0)
    merge4_1 = concatenate([conv4_0, up4_1], axis=3)
    conv4_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4_1)
    conv4_1 = BatchNormalization()(conv4_1)
    
    up3_2 = UpSampling2D(size=(2, 2))(conv4_1)
    merge3_2 = concatenate([conv3_0, up3_2], axis=3)
    conv3_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3_2)
    conv3_2 = BatchNormalization()(conv3_2)
    
    up2_3 = UpSampling2D(size=(2, 2))(conv3_2)
    merge2_3 = concatenate([conv2_0, up2_3], axis=3)
    conv2_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2_3)
    conv2_3 = BatchNormalization()(conv2_3)

    up1_4 = UpSampling2D(size=(2, 2))(conv2_3)
    merge1_4 = concatenate([conv1_0, up1_4], axis=3)
    conv1_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1_4)
    conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Dropout(0.5)(conv1_4)

    # Output layer
    conv_out = Conv2D(1, 1, activation='sigmoid')(conv1_4)

    model = tf.keras.Model(inputs=inputs, outputs=conv_out)
    return model
