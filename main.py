from utils import *

import tensorflow as tf
import os

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


print(f"Tensorflow version: {tf.__version__}")

# LOAD DATASET
dataset_dir = os.path.join(os.path.dirname("./data/"), 'stack_overflow_16k')
print(os.listdir(dataset_dir))  # Files/Folders contained in directory

# GET TRAINING DATA
seed = 42
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    validation_split=0.2,
    subset='training',
    seed=seed)  # Optional random seed for shuffling and transformations. Thus, we can reproduce the results when using
                # random generators. That is, using the same seed two times, results in the same random number.

# print(type(raw_train_ds))  # BatchDataset
# print(type(raw_train_ds.take(1)))  # TakeDataset

# Exploring the dataset
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(2):
        print("Question", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])
# Show classes
print(raw_train_ds.class_names)

# GET VALIDATION DATA
raw_validation_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    validation_split=0.2,
    subset='validation',
    seed=seed)

# GET TESTING DATA
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(os.path.join(dataset_dir, "test"))


# PREPARE THE DATASET FOR TRAINING
vocabulary_size = 10000
max_question_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocabulary_size,
    output_mode='int',
    output_sequence_length=max_question_length)
