import torch
from PIL import Image
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import cv2

import torch.nn.functional as F
from torch import nn, Tensor
from detectron2.engine.defaults import create_ddp_model
from detrex.modeling import ema
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)


def main(args):
    cfg = LazyConfig.load('/home/stefan/Quest4FMR/configs/ovdino_base_eval_coco.py')
    cfg_model = "/home/stefan/Quest4FMR/configs/models/ov_dino/ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth"
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #model_path = 'https://huggingface.co/hao9610/OV-DINO/resolve/main/ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth'
    # Load the state_dict from the URL
    #state_dict = torch.hub.load_state_dict_from_url(model_path)
    model = instantiate(cfg_model)
    model.to(device)
    model = create_ddp_model(model)

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    #model_path = 'https://huggingface.co/hao9610/OV-DINO/resolve/main/ovdino_swint_og-coco50.6_lvismv39.4_lvis32.2.pth'
    # Load the state_dict from the URL
    #state_dict = torch.hub.load_state_dict_from_url(model_path)

    model = instantiate(cfg.model)
    # Load the state_dict into the model
    model.load_state_dict(state_dict)


    model.to(device)
    model = create_ddp_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation

    vl_model = clip_vit_b16(pretrained=True)
    vl_model = vl_model.to(device)

    text_transform = CLIPTextTransform()
    torch_transform = th_transforms.Compose([
        th_transforms.Resize(im_size),
        th_transforms.ToTensor(),
        th_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_path = "images/query_1.png"
    with Image.open(img_path) as im:
        im_og = (np.array(im) * (1 / 255))
        o_height, o_width, _ = im_og.shape
        print(o_height, o_width)
        image_batch = torch_transform(im)
        image_batch = torch.unsqueeze(image_batch, 0).to(device)

    #text = ["cat", "toy", "clutter"]
    #text = ["toy cat"]
    text = ["This is a toy cat."]
    #text = ["A plane."]

    text_batch = text_transform(text)
    print('text_t: ', text_batch)

    img_embedding, txt_embedding = vl_model(image_batch, text_batch.to(device))
    print('embeddings: ', img_embedding.shape, txt_embedding.shape)

    img_emb = img_embedding.detach().cpu()
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
    args = default_argument_parser().parse_args()
    main(args)