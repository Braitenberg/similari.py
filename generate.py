import os
import cv2
import sys
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

from tqdm import tqdm
from random import shuffle
from os import path
from sklearn.preprocessing import OneHotEncoder

IMG_SIZE = 150
MAIN_DIR = 'static/flowers'
EMBEDDING_DIM = 2048

# expected folder structure:
# MAIN_DIR
# |
#  -- category
#     |
#     -- image
#     -- image
#     -- image
#  -- category
#     |
#     -- image
#     -- image
#     -- image

### CREATE MODEL ###
input_dimensions = (IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.applications.ResNet50(weights=None, input_shape=input_dimensions, classes=5)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

### LOAD DATA ###
images = []
labels = []

categories = os.listdir(MAIN_DIR)

for category in categories:
    location = f'{MAIN_DIR}/{category}'
    print(category)
    for image_path in tqdm(os.listdir(location)):
            img = cv2.imread(f'{location}/{image_path}' ,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            images.append(np.array(img))
            labels.append(category)

# fig,ax=plt.subplots(5,2)
# fig.set_size_inches(15,15)
# for i in range(5):
#     for j in range (2):
#         index = random.randint(0, len(images))
#         ax[i,j].imshow(images[index])
#         ax[i,j].set_title('Flower: '+labels[index])
        
# plt.tight_layout()
# plt.show()

### PRE-PROCESS DATA ###
train_cut = round(len(images) / 4)
test_cut = len(images) - train_cut

labeled_images = list(zip(images, labels))

shuffle(labeled_images)

### ENCODE DATA ###
images, labels = zip(*labeled_images)

enc = OneHotEncoder(sparse=False)

encoded_labels = enc.fit_transform(np.array(labels).reshape(-1,1))

train_images = np.array(images[:train_cut])
train_labels = np.array(encoded_labels[:train_cut])

test_images  = np.array(images[-test_cut:])
test_labels  = np.array(encoded_labels[-test_cut:])

### TRAIN OR LOAD MODEL
tf_model_path = "tf_predictor"

if (path.exists(tf_model_path) or len(sys.argv) > 1):
    model = tf.keras.models.load_model(tf_model_path)
else:
    model.fit(train_images, train_labels, epochs=20)
    model.save(tf_model_path)


# Replace the output with an embedding layer
feature_generator = tf.keras.layers.Embedding(output_dim=EMBEDDING_DIM)(
    model.layers[-2].output
)

tf.keras.Model(
    inputs = model.input,
    outputs=feature_generator
).save("tf_featurator") # featurator returns x arbitrary `features`
