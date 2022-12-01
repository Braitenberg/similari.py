import cv2
import PIL
import io
import numpy as np
import tensorflow as tf
from flask import (
  Flask,
  render_template,
  request,
  jsonify
)
from pymilvus import (
    connections,
    Collection,
)

app = Flask(__name__)

EMBEDDER = tf.keras.models.load_model('tf_featurator')

@app.after_request
def apply_caching(response):
    response.headers["Cache-Control"] = "no-cache"
    return response

@app.route("/")
def index():
    return render_template('form.html')

@app.route("/similarity", methods=["POST"])
def find_similair_images():
  connections.connect("default", host="localhost", port="19530")
  embeddings = embeddings_for_image(request.files['image'])
  collection = Collection("images")
  collection.load()
  results = collection.search(
    data = [embeddings], 
    anns_field = "embeddings", 
    limit = 5, 
    expr = None,
    param = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5},
    consistency_level = "Strong",
    output_fields = ["image_path"]
  )

  return jsonify([hit.entity.get('image_path') for hit in results[0]])

def embeddings_for_image(image):
  picture_stream = io.BytesIO(image.read())
    
  img = PIL.Image.open(picture_stream)
  img = np.array(img)
  img = cv2.resize(img, (150,150))

  embeddable = np.expand_dims(img, axis=0)

  return EMBEDDER.predict(embeddable, verbose="default")[0].tolist()
