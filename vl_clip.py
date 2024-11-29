import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import cv2


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation

    img_path = "images/query_0.png"
    with Image.open(img_path) as im:
        im_og = (np.array(im) * (1 / 255))
        o_height, o_width, _ = im_og.shape
        #print(o_height, o_width)
        #image_batch = torch_transform(im)
        #image_batch = torch.unsqueeze(image_batch, 0).to(device)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=["a cat"], images=im_og, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    print("loop through model outputs")
    for idx, whatever in enumerate(outputs):
        print(idx, whatever)

    vision_outputs = outputs.vision_model_output.last_hidden_state

    print(vision_outputs.shape)
    img_emb = vision_outputs.detach().cpu()
    img_emb = img_emb[:, 1:, :]
    b, tokens, feat = img_emb.shape
    # img_emb = img_emb.view(b, int(math.sqrt(tokens)), int(math.sqrt(tokens)), feat)
    img_emb = img_emb.detach().cpu()

    # stupid projection to image space
    pca = PCA(n_components=3, svd_solver='full')
    img_viz = img_emb[0, ...]
    img_viz = pca.fit_transform(img_viz)
    # img_viz = pca.transform(img_viz)
    img_viz = np.reshape(img_viz, (int(math.sqrt(tokens)), int(math.sqrt(tokens)), 3))
    print(img_viz.shape)

    img_viz = cv2.resize(img_viz, dsize=(o_height, o_width), interpolation=cv2.INTER_CUBIC)
    img_viz[..., 0] = (img_viz[..., 0] - np.nanmin(img_viz[..., 0])) / np.max(
        img_viz[..., 0] - np.nanmin(img_viz[..., 0]))  # * (255 / np.max(img_viz[..., 0]))
    img_viz[..., 1] = (img_viz[..., 1] - np.nanmin(img_viz[..., 1])) / np.max(
        img_viz[..., 1] - np.nanmin(img_viz[..., 1]))  # * (255 / np.max(img_viz[..., 1]))
    img_viz[..., 2] = (img_viz[..., 2] - np.nanmin(img_viz[..., 2])) / np.max(
        img_viz[..., 2] - np.nanmin(img_viz[..., 2]))  # * (255 / np.max(img_viz[..., 2]))

    print("im_og: ", np.min(im_og), np.max(im_og))
    print("im_emb: ", np.min(img_viz), np.max(img_viz))
    img_comp = np.concatenate([im_og, img_viz], axis=1)
    plt.imshow(img_comp, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main()