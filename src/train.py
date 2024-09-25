import os
import tensorflow as tf
from model import create_model
from utils.data_preprocessing import load_data

train_generator, test_generator = load_data('data/train', 'data/test')

model = create_model()
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the model
model.save('models/model.h5')
