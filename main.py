from utils import *

import os
import numpy as np

from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


print(f"Tensorflow version: {tf.__version__}")

############################## 1. LOAD DATASET ##############################

dataset_dir = os.path.join(os.path.dirname("./data/"), 'stack_overflow_16k')
# print(os.listdir(dataset_dir))  # Files/Folders contained in directory

# GET TRAINING DATA
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    validation_split=0.2,
    subset='training',
    seed=seed)  # Optional random seed for shuffling and transformations. Thus, we can reproduce the results when using
                # random generators. That is, using the same seed two times, results in the same random number.

# print(type(raw_train_ds))  # BatchDataset: Dataset of batches/groups (32 elements by default in the function above)
# print(type(raw_train_ds.take(1)))  # TakeDataset: Returns n (1 in this case) batches from the BatchDataset

# GET VALIDATION DATA
raw_validation_ds = tf.keras.preprocessing.text_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    validation_split=0.2,
    subset='validation',
    seed=seed)

# GET TESTING DATA
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(os.path.join(dataset_dir, "test"))

# EXPLORE THE DATASET
# for text_batch, label_batch in raw_train_ds.take(1):
#     print("Question", text_batch.numpy()[0])
#     print("Label", label_batch.numpy()[0])
# Show classes
# print(raw_train_ds.class_names)  # ['csharp', 'java', 'javascript', 'python']

#################### 2. PREPARE THE DATASET FOR TRAINING ####################

vocabulary_size = 5000
max_question_length = 500

vectorize_layer = TextVectorization(
    # standardize=custom_standardization,  # In this case, this standardization function doesn't seem to improve the performance
    max_tokens=vocabulary_size,
    output_mode='int',  # Create unique integer indices for each token
    output_sequence_length=max_question_length)

train_text = raw_train_ds.map(lambda x, y: x)  # Make a text-only dataset (WITHOUT labels)
vectorize_layer.adapt(train_text)  # This will analyze the dataset, determine the frequency of individual string values,
                                   # and create a 'vocabulary' from them

# print(vectorize_layer.get_vocabulary())  # Look up the relationship between tokens (strings) and integers

# APPLAY THE TextVectorization LAYER TO EACH DATASET
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_validation_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# EXPLORE THE DATASET
# for text_batch, label_batch in train_ds.take(1):
#     print("Question", text_batch.numpy()[0])
#     print("Label", label_batch.numpy()[0])

################# 3. CONFIGURE THE DATASET FOR PERFORMANCE ##################

'''
Note: Two important methods.
- .cache()

Keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while 
training your model. If your dataset is too large to fit into memory, you can also use this method to create a 
performant on-disk cache, which is more efficient to read than many small files.

- .prefetch()

Overlaps data preprocessing and model execution while training.
'''

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# It seems to improve the computational time in the following epochs

############################ 4. CREATE THE MODEL ############################

embbeding_dim = 128

model = tf.keras.Sequential([
    # This layer takes the integer-encoded text and looks up an embedding vector for each word-index. These vectors are learned as the model trains.
    Embedding(input_dim=vocabulary_size + 1,  # Size of the vocabulary (i.e. maximum integer index + 1)
              output_dim=embbeding_dim,
              input_length=max_question_length),  # Length of input sequences, when it is constant
    Dropout(rate=0.2),

    # For each feature dimension, this layer takes average among all time steps
    GlobalAveragePooling1D(),
    Dropout(0.2),

    # Fully connected layer
    Dense(units=4)
])

'''
Note: 
If you want to detect the presence of something in your sequences, MAX POOLING is a good option.
If the contribution of the entire sequence seems important to your result, then AVERAGE POOLING sounds reasonable.
'''

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),  # If this was a binary classification problem --> Use BinaryCrossentropy
    optimizer='adam',
    metrics=['accuracy'])

###################### 5. TRAIN AND EVALUATE THE MODEL ######################

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

plot_loss_acc(range(1, epochs +1), history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])

############################# 6. EXPORT MODEL ##############################

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    Activation('sigmoid')
])

export_model.compile(loss=SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

loss, accuracy = export_model.evaluate(raw_test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

########################## 7. TEST WITH NEW DATA ###########################

result = export_model.predict(["blank - get class of type from imported module i am trying to import a class from a file with a dynamic name. here is my file importer:..def get_channel_module(app, method):.    mod_name = 'channel.%s.%s' % (app, method).    module = __import__(mod_name, globals(), locals(), [method]).    return module...this imports the specific blank file, for example, some_file.py, which looks like this:..class someclassa(baseclass):.    def __init__(self, *args, **kwargs):.        return..class someclassb():.    def __init__(self, *args, **kwargs):.        return...what i want to do is return only the class which extends baseclass from the imported file, so in this instance, someclassa. is there any way to do this?"])
print(result)  # Should be Python (idx: 3)

print(f"Programming Language: {raw_train_ds.class_names[np.argmax(result[0])]}")
