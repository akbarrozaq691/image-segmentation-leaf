import cv2
#library untuk mrcnn
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import model as modellib, utils
from tools import *
import skimage

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir='')
model_path = 'model/mask_rcnn_leafandbranch_0023.h5'

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir='')

dataset_train = CocoLikeDataset()
dataset_train.load_data("dataset/train/train-categories.json","dataset/train/images")
dataset_train.prepare()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

window_name = 'Image'

# gambar = cv2.imread('dataset/train/images/Leaf_001.jpg')
# gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
gambar = skimage.io.imread('dataset/train/images/Leaf_001.jpg')
results = model.detect([gambar], verbose=1)
# Visualize the results
r = results[0]
masks = r["masks"]
class_ids = r["class_ids"]
scores = r["scores"]
boxes = r['rois']

visualize.display_instances(gambar, boxes, masks, class_ids, 
                                dataset_train.class_names, scores, figsize=(8, 8))