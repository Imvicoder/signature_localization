import cv2
import numpy as np
from keras.utils import Sequence
from utils import BoundBox, bbox_iou

class Generator(Sequence):
	def __init__(self, images, config, shuffle=True, norm=None):
		"""
		params: images: list of dictionary of cheque image and corresponding signatures
				config: generator config	
				norm : function to normalise the image

		return None
		"""

		self.images = images
		self.config = config
		self.shuffle = shuffle
		self.norm = norm

		self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]
		
		if shuffle:
			np.random.shuffle(self.images)


	def __len__(self):
		return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

	def read_data(self, train_instance):
		img_path = train_instance["image_path"]
		img = cv2.imread(img_path)
		objects = train_instance["signatures"]
		return img, objects

	def __getitem__(self, idx):
		l_bound = idx * self.config['BATCH_SIZE']
		r_bound = (idx + 1) * self.config['BATCH_SIZE']

		if r_bound > len(self.images):
			r_bound = len(self.images)
			l_bound = r_bound - self.config['BATCH_SIZE']

		instance_count = 0

		x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
		b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
		y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

		for train_instance in self.images[l_bound:r_bound]:
			img, all_objects = self.read_data(train_instance)

			true_box_index = 0

			for obj in all_objects:
				center_x = (obj['sig_x'] / (float(self.config['IMAGE_W'])) * self.config['GRID_W'])
				center_y = (obj['sig_y'] / (float(self.config['IMAGE_H'])) * self.config['GRID_H'])

				grid_x = int(np.floor(center_x))
				grid_y = int(np.floor(center_y))

				width = (obj['sig_w'] / (float(self.config['IMAGE_W'])) * self.config['GRID_W'])
				height = (obj['sig_h'] / (float(self.config['IMAGE_H'])) * self.config['GRID_H'])

				box = [center_x, center_y, width, height]

				 # find the anchor that best predicts this box
				best_anchor = -1
				max_iou     = -1

				shifted_box = BoundBox(0, 0, width, height)

				for i in range(len(self.anchors)):
					anchor = self.anchors[i]
					iou = bbox_iou(shifted_box, anchor)

					if max_iou < iou:
						best_anchor = i
						max_iou = iou

				# assign ground truth x, y, w, h, confidence and class probs to y_batch
				y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
				y_batch[instance_count, grid_y, grid_x, best_anchor,  4 ] = 1.
				y_batch[instance_count, grid_y, grid_x, best_anchor, 5] = 1
				
				# assign the true box to b_batch
				b_batch[instance_count, 0, 0, 0, true_box_index] = box
				
				true_box_index += 1
				true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

			# assign input image to x_batch
			if self.norm != None: 
				x_batch[instance_count] = self.norm(img)

			instance_count += 1

		return [x_batch, b_batch], y_batch

	def on_epoch_end(self):
		if self.shuffle: np.random.shuffle(self.images)



