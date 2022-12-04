import os
import cv2
import PIL
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from pymilvus import (
    utility,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)



# https://github.com/paucarre/tiefvision

IMG_SIZE = 150
MAIN_DIR = 'static/flowers'

model = tf.keras.models.load_model('tf_featurator')

model.summary()

images      = []
image_paths = []

i=0

for category in os.listdir(MAIN_DIR):
  location = f'{MAIN_DIR}/{category}'
  print(category)
  for image_path in tqdm(os.listdir(location)):
    if i > 100:
      break

    img = open(f'{location}/{image_path}' , "rb")
    img = PIL.Image.open(img)
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    images.append(np.array(img))
    image_paths.append(f'{location}/{image_path}')
    # i = i+1

### GENERATE RESULTS
embeddings = []

for image in tqdm(images):
  input = np.expand_dims(image, axis=0)
  result = model.predict(input, verbose="default")
  embeddings.append(result[0].tolist())

### PUSH RESULTS TO MILVUS
connections.connect("default", host="localhost", port="19530")

fields = [
  FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
  FieldSchema(
    name="image_path", 
    dtype=DataType.VARCHAR, 
    max_length=200,
  ),
  FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=model.output.shape[1])
]

schema = CollectionSchema(fields)

utility.drop_collection("images")

collection = Collection("images", schema)
collection.create_index(
  field_name="embeddings",
  index_params={
    "metric_type":"L2",
    "index_type":"IVF_FLAT",
    "params":{"nlist":1024}
})

collection.insert([image_paths, embeddings])
