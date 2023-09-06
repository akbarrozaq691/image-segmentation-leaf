#library untuk operasional
import os
import sys

#library untuk model
from flask import Flask, render_template, request
import cv2
import numpy as np
from tools import *

#library untuk mrcnn
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import model as modellib, utils
import skimage


app = Flask(__name__)

#route utama
@app.route("/")
def home():
    return render_template('index.html')

#API untuk prediksi gambar
@app.route("/predict", methods = ['POST'])
def predict():
    dataset_train = CocoLikeDataset()
    dataset_train.load_data("dataset/train/annotations.json","dataset/train/images")
    dataset_train.prepare()

    # Load the image
    image_data = request.files["image"].read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir='')
    model_path = 'model/mask_rcnn_leafandbranch_0007.h5'

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Run inference
    results = model.detect([image], verbose=1)

    # Visualize the results
    r = results[0]
    masks = r["masks"]
    class_ids = r["class_ids"]
    scores = r["scores"]
    boxes = r['rois']

    visualize.display_instances(image, boxes, masks, class_ids,
                                dataset_train.class_names, scores, figsize=(8, 8))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)