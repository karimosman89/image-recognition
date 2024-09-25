import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, img_size=(64, 64)):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        class_mode='sparse',
        batch_size=32
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        class_mode='sparse',
        batch_size=32
    )

    return train_generator, test_generator
