from collections import Counter
import numpy as np
import pickle as pkl
from generator import Generator

from yolo_network import YOLO

data = pkl.load(open("SIG_LABELLED_DATA.pkl", "rb")) 

images = data["signature_data"]
np.random.shuffle(images)
train_imgs = images[:1400]
valid_imgs = images[1400:]

input_size = 416
labels = ["signature"]
max_box_per_image = 4
anchors = [2.14,1.52, 2.72,1.05, 2.73,2.15, 3.33,1.42, 3.90,2.66, 3.94,1.79, 4.33,1.07, 5.32,1.42, 5.54,2.08, 5.82,3.28]
train_times = 8
valid_times = 1
nb_epochs = 10
learning_rate = 1e-4
batch_size = 16
warmup_epochs = 3
object_scale = 5.0
no_object_scale = 1.0
coord_scale = 1.0
class_scale = 1.0
debug = True


yolo = YOLO(input_size, labels, max_box_per_image, anchors)

yolo.train(train_imgs         = train_imgs,
		   valid_imgs         = valid_imgs,
		   train_times        = train_times,
		   valid_times        = valid_times,
		   nb_epochs          = nb_epochs, 
		   learning_rate      = learning_rate, 
		   batch_size         = batch_size,
		   warmup_epochs      = warmup_epochs,
		   object_scale       = object_scale,
		   no_object_scale    = no_object_scale,
		   coord_scale        = coord_scale,
		   class_scale        = class_scale,
		   saved_weights_name = "signature_localisation_weights.h5",
		   debug              = debug)