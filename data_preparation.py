import json
import cv2
import pickle as pkl

# cheques

SIG_IMAGE_PATH1 = "/home/vikas/object_detection/signature/chunk_1_tariq/"
SIG_IMAGE_PATH2 = "/home/vikas/object_detection/signature/chunk_2_taufiq/"
SIG_IMAGE_PATH3 = "/home/vikas/object_detection/signature/chunk_3_tarique/"
SIG_IMAGE_PATH4 = "/home/vikas/object_detection/signature/chunk_4_taufique/"
SIG_IMAGE_PATH5 = "/home/vikas/object_detection/signature/chunk_5_tariq/"
SIG_IMAGE_PATH6 = "/home/vikas/object_detection/signature/chunk_6_taufiq/"
SIG_IMAGE_PATH7 = "/home/vikas/object_detection/signature/chunk_7_tariq/"
SIG_IMAGE_PATH8 = "/home/vikas/object_detection/signature/chunk_8_taufiq/"
SIG_IMAGE_PATH9 = "/home/vikas/object_detection/signature/chunk_9_tariq/"
SIG_IMAGE_PATH10 = "/home/vikas/object_detection/signature/chunk1_only_sig_taufique/"
SIG_IMAGE_PATH11 = "/home/vikas/object_detection/signature/chunk2_only_sig_tarique/"

SIG_LABEL_JSON1 = "/home/vikas/object_detection/signature/chunk_1_tariq.json"
SIG_LABEL_JSON2 = "/home/vikas/object_detection/signature/chunk_2_taufiq.json"
SIG_LABEL_JSON3 = "/home/vikas/object_detection/signature/chunk_3_tarique.json"
SIG_LABEL_JSON4 = "/home/vikas/object_detection/signature/chunk_4_taufique.json"
SIG_LABEL_JSON5 = "/home/vikas/object_detection/signature/chunk_5_tariq.json"
SIG_LABEL_JSON6 = "/home/vikas/object_detection/signature/chunk_6_taufiq.json"
SIG_LABEL_JSON7 = "/home/vikas/object_detection/signature/chunk_7_tariq.json"
SIG_LABEL_JSON8 = "/home/vikas/object_detection/signature/chunk_8_taufiq.json"
SIG_LABEL_JSON9 = "/home/vikas/object_detection/signature/chunk_9_tariq.json"
SIG_LABEL_JSON10 = "/home/vikas/object_detection/signature/chunk1_only_sig_taufique.json"
SIG_LABEL_JSON11 = "/home/vikas/object_detection/signature/chunk2_only_sig_tarique.json"

# only sig
SIG_IMAGE_PATH12 = "/home/vikas/object_detection/signature/chunk6_only_sig_taufique/"
SIG_IMAGE_PATH13 = "/home/vikas/object_detection/signature/chunk7_only_sig_tarique/"
SIG_IMAGE_PATH14 = "/home/vikas/object_detection/signature/chunk8_only_sig_taufique/"
SIG_IMAGE_PATH15 = "/home/vikas/object_detection/signature/chunk9_only_sig_taufique/"


SIG_LABEL_JSON12 = "/home/vikas/object_detection/signature/chunk6_only_sig_taufique_label.json"
SIG_LABEL_JSON13 = "/home/vikas/object_detection/signature/chunk7_only_sig_tarique.json"
SIG_LABEL_JSON14 = "/home/vikas/object_detection/signature/chunk8_sig_only_taufique_label.json"
SIG_LABEL_JSON15 = "/home/vikas/object_detection/signature/chunk9_only_sig_taufique_label.json"

def parse_json(SIG_LABEL_JSON, SIG_IMAGE_PATH, signature_data):
	data1 = json.load(open(SIG_LABEL_JSON, "r"))
	filenames = [data1["_via_img_metadata"][i]["filename"] for i in data1["_via_img_metadata"]]
	regions = [data1["_via_img_metadata"][i]["regions"] for i in data1["_via_img_metadata"]]


	for filename, region in zip(filenames,regions):
		print(f"filename: {filename} and region: {region}")
		image_path = f"{SIG_IMAGE_PATH}{filename}"
		sig_data = {}

		img = cv2.imread(image_path)
		if img is None:
			print(f"failed image_path: {image_path}")
			continue
		img_height, img_width = img.shape[0], img.shape[1]
		sig_data["image_path"] = image_path
		sig_data["img_height"] = img_height
		sig_data["img_width"] = img_width

		signatures = []
		for i in range(len(region)):
			sig = {}
			sig_x = region[i]["shape_attributes"]["x"]
			sig_y = region[i]["shape_attributes"]["y"]
			sig_w = region[i]["shape_attributes"]["width"]
			sig_h = region[i]["shape_attributes"]["height"]

			sig["sig_x"] = sig_x
			sig["sig_y"] = sig_y
			sig["sig_w"] = sig_w
			sig["sig_h"] = sig_h
			signatures.append(sig)

		sig_data["signatures"] = signatures

		print(f"sig_data: {sig_data}")
		signature_data.append(sig_data)

signature_data = []
parse_json(SIG_LABEL_JSON1, SIG_IMAGE_PATH1, signature_data)
parse_json(SIG_LABEL_JSON2, SIG_IMAGE_PATH2, signature_data)
parse_json(SIG_LABEL_JSON3, SIG_IMAGE_PATH3, signature_data)
parse_json(SIG_LABEL_JSON4, SIG_IMAGE_PATH4, signature_data)
parse_json(SIG_LABEL_JSON5, SIG_IMAGE_PATH5, signature_data)
parse_json(SIG_LABEL_JSON6, SIG_IMAGE_PATH6, signature_data)
parse_json(SIG_LABEL_JSON7, SIG_IMAGE_PATH7, signature_data)
parse_json(SIG_LABEL_JSON8, SIG_IMAGE_PATH8, signature_data)
parse_json(SIG_LABEL_JSON9, SIG_IMAGE_PATH9, signature_data)
parse_json(SIG_LABEL_JSON10, SIG_IMAGE_PATH10, signature_data)
parse_json(SIG_LABEL_JSON11, SIG_IMAGE_PATH11, signature_data)

# parse_json(SIG_LABEL_JSON12, SIG_IMAGE_PATH12, signature_data)
# parse_json(SIG_LABEL_JSON13, SIG_IMAGE_PATH13, signature_data)
# parse_json(SIG_LABEL_JSON14, SIG_IMAGE_PATH14, signature_data)
# parse_json(SIG_LABEL_JSON15, SIG_IMAGE_PATH15, signature_data)


SIG_LABELLED_DATA = {}
SIG_LABELLED_DATA["signature_data"] = signature_data
SIG_LABELLED_DATA["signature"] = len(signature_data)
print(len(SIG_LABELLED_DATA["signature_data"]))

# pkl.dump(SIG_LABELLED_DATA, open("ONLY_FIN_SIG_LABELLED_DATA.pkl", "wb"))
pkl.dump(SIG_LABELLED_DATA, open("SIG_LABELLED_DATA.pkl", "wb"))

# for filename in filenames:
# 	sig_data = {}
# 	image_path = f"{SIG_IMAGE_PATH1}{filename}"
# 	for i in data1["_via_img_metadata"]:
# 		for j in data1["_via_img_metadata"]

