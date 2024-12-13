import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import cv2

import torch.nn.functional as F
from torch import nn, Tensor


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation

    img_path = "images/query_2.png"
    with Image.open(img_path) as im:
        im_og = (np.array(im) * (1 / 255))
        o_height, o_width, _ = im_og.shape
        #print(o_height, o_width)
        #image_batch = torch_transform(im)
        #image_batch = torch.unsqueeze(image_batch, 0).to(device)

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    inputs = processor(images=im_og, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    print(attentions)

    for i, lay in enumerate(attentions):
        print(i, lay.shape)

    last_hidden_states = outputs[0]

    # We have to force return_dict=False for tracing
    model.config.return_dict = False

    with torch.no_grad():
        traced_model = torch.jit.trace(model, [inputs.pixel_values])
        traced_outputs = traced_model(inputs.pixel_values)

    print((last_hidden_states - traced_outputs[0]).abs().max())

    img_emb = last_hidden_states.detach().cpu()
    img_emb = img_emb[:, 1:, :]
    b, tokens, feat = img_emb.shape
    # img_emb = img_emb.view(b, int(math.sqrt(tokens)), int(math.sqrt(tokens)), feat)
    img_emb = img_emb.detach().cpu()

    # stupid projection to image space
    pca = PCA(n_components=3, svd_solver='full')
    img_viz = img_emb[0, ...]
    img_viz = pca.fit_transform(img_viz)
    # img_viz = pca.transform(img_viz)
    print(img_viz.shape)
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