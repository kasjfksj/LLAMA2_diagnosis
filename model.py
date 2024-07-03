import transformers
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel
from data import get_data
Img_encoder = transformers.CLIPImageProcessor(do_rescale=False)
transform = transforms.ToTensor()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def image_proccessor(img_url):
    tensor = transform(Image.open('img_url'))
    a = torch.tensor(np.array(Img_encoder.preprocess(tensor)["pixel_values"]))
    return  model.get_image_features(a)

text, img_feature = get_data()
print(len(img_feature[1]))
class Linearprobe(torch.nn.Module):

    def __init__(self):
        super(Linearprobe, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    


