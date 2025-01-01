import tensorflow as tf
from tensorflow.keras import layers, models

def fcn_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Convolutional block 1
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Giảm dropout
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Convolutional block 2
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Giảm dropout
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Convolutional block 3
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Giảm dropout
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Convolutional block 4
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Upsampling path (tăng số lượng layers)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)


    # Final output layer
    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Create model
    model = models.Model(inputs, x)

    return model
