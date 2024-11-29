import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import torch.nn.functional as F
import cv2

from torchmultimodal.models.coca.coca_model import coca_vit_b_32
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
from torchmultimodal.utils.common import load_module_from_url

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation
    #vl_model = ClipBackbones("vit_b16", im_size)
    vl_model = coca_vit_b_32()
    vl_model = vl_model.to(device)

    torch_transform = th_transforms.Compose([
        th_transforms.Resize(im_size),
        th_transforms.ToTensor(),
        th_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_path = "/home/stefan/Quest4FMR/images/query_1.png"
    with Image.open(img_path) as im:
        im_og = (np.array(im) * (1 / 255))
        o_height, o_width, _ = im_og.shape
        print(o_height, o_width)
        image_batch = torch_transform(im)
        image_batch = torch.unsqueeze(image_batch, 0).to(device)

    embedding = vl_model.vision_encoder(image_batch)
    embedding = embedding.last_hidden_state#[:, :, :]
    img_emb = embedding.detach().cpu()
    b, tokens, feat = img_emb.shape

    #image_batch = torch.unsqueeze(tensor_image, 0)

    # The image is now a PyTorch tensor
    #print(image_batch.shape)

    pca = PCA(n_components=3, svd_solver='full')
    img_viz = img_emb[0, ...]
    img_viz = pca.fit_transform(img_viz)
    # img_viz = pca.transform(img_viz)
    print(img_viz.shape)
    img_viz = np.reshape(img_viz, (int(math.sqrt(tokens)), int(math.sqrt(tokens)), 3))

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