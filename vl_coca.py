import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
import numpy as np

import torch.nn.functional as F

from torchmultimodal.models.clip.model import CLIP
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
from torchmultimodal.utils.common import load_module_from_url


CLIP_MODEL_MAPPING = {
    "vit_b16": "https://download.pytorch.org/models/multimodal/clip/clip_vit_b16.pt",
    "vit_b32": "https://download.pytorch.org/models/multimodal/clip/clip_vit_b32.pt",
    "vit_l14": "https://download.pytorch.org/models/multimodal/clip/clip_vit_l14.pt",
    "rn50": "https://download.pytorch.org/models/multimodal/clip/clip_rn50.pt",
    "rn101": "https://download.pytorch.org/models/multimodal/clip/clip_rn101.pt",
    "rn50x4": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x4.pt",
    "rn50x16": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x16.pt",
    "rn50x64": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x64.pt",
}


class ClipOverloaded(CLIP):
    """CLIP is a model for contrastive pretraining between two modalities."""

    def __init__(self, encoder_a: nn.Module,
        encoder_b: nn.Module, image_size):
        super(CLIP, self).__init__()

        self.encoder_a = encoder_a
        self.encoder_b = encoder_b

        self.torch_transform = th_transforms.Compose([
            th_transforms.Resize((image_size, image_size)),
            th_transforms.ToTensor(),
            th_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(
        self,
        features_a: torch.Tensor,
    ):

        embeddings_a = self.encoder_a(features_a)
        embeddings_a = F.normalize(embeddings_a)
        return embeddings_a

    def transform_img(self, image):
        return self.torch_transform(image)


def clip_vit_b16(pretrained: bool = False) -> ClipOverloaded:
    vision_encoder = CLIPViTEncoder(
        image_size=224, patch_size=16, layers=12, heads=12, width=768, embedding_dim=512
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    clip = ClipOverloaded(vision_encoder, text_encoder, 224)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["vit_b16"])
    return clip


def clip_rn50(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=32,
        width=64,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    clip = ClipOverloaded(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn50"])
    return clip



class ClipBackbones(CLIP):
    """CLIP is a model for contrastive pretraining between two modalities.

    Inputs:
    """

    def __init__(self, backbone, image_size):
        super(self).__init__()

        if backbone == "vit_b16":
            self.model = clip_vit_b16(pretrained=True)
        elif backbone == "vit_l14":
            from torchmultimodal.models.clip.model import clip_vit_b32
            self.model = clip_vit_b32(pretrained=True)
        elif backbone == "vit_l14":
            from torchmultimodal.models.clip.model import clip_vit_l14
            self.model = clip_vit_l14(pretrained=True)
        elif backbone == "rn50":
            from torchmultimodal.models.clip.model import clip_rn50
            self.model = clip_rn50(pretrained=True)
        elif backbone == "rn101":
            from torchmultimodal.models.clip.model import clip_rn101
            self.model = clip_rn101(pretrained=True)
        else:
            print("Backbone name not valid. Valid model names are [vit_b16, vit_b32, vit_l14, rn50, rn101] ")

        self.im_size = image_size
        self.torch_transform = th_transforms.Compose([
            th_transforms.Resize(self.im_size),
            th_transforms.ToTensor(),
            th_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, image):
        #img = self.transform_img(image)
        return self.model(image)

    def transform_img(self, image):
        return self.torch_transform(image)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation
    #vl_model = ClipBackbones("vit_b16", im_size)
    vl_model = clip_vit_b16(pretrained=True)

    img_path = "/home/stefan/Quest4FMR/images/query_0.png"
    with Image.open(img_path) as im:
        print(im.size)
        query_im = im.convert('RGB')
        #query_im = np.array(im)

    print(query_im.size)

    torch_img = vl_model.transform_img(query_im)
    print(torch_img.shape)
    image_batch = torch.unsqueeze(torch_img, 0)
    embedding = vl_model(image_batch)

    #image_batch = torch.unsqueeze(tensor_image, 0)

    # The image is now a PyTorch tensor
    #print(image_batch.shape)

    img_emb = embedding.detach().cpu()

    print(img_emb.shape)

if __name__ == "__main__":
    main()