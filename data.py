import pandas as pd
from PIL import Image
import os
import transformers
import torch
from transformers import CLIPModel

import torchvision.transforms as transforms
import numpy as np

'''encoding images'''
Img_encoder = transformers.CLIPImageProcessor(do_rescale=False)
transform = transforms.ToTensor()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

'''helper function'''
def image_proccessor(img_url):
    tensor = transform(Image.open(img_url))
    a = torch.tensor(np.array(Img_encoder.preprocess(tensor)["pixel_values"]))
    return model.get_image_features(a)

data = pd.read_csv("dataset/HAM10000_metadata.csv")
del(data["lesion_id"])
data = data.values.tolist()
img_feature = []
classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}
for i in range(len(data)):
    data[i].append(image_proccessor("dataset/ham_images/"+data[i][0]+".jpg"))
def get_data():
    return data