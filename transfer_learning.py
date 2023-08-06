import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_transfer_model(num_classes):
    """
    Create a transfer learning model based on VGG16 with custom output layers.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Transfer learning model.
    """
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    # Load your custom dataset for transfer learning
    # Replace 'x_train', 'y_train', 'x_val', 'y_val' with your dataset

    num_classes = 10  # Replace with the number of output classes in your custom dataset

    model = create_transfer_model(num_classes)

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model with data augmentation
    datagen = ImageDataGenerator(rescale=1.0/255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

    batch_size = 32
    epochs = 10

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(x_train) // batch_size,
              validation_data=(x_val, y_val),
              epochs=epochs)

    # Save the model for future use
    model.save('transfer_model.h5')

if __name__ == "__main__":
    main()
