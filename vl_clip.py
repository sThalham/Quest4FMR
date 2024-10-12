import torch
from PIL import Image
from torchvision import transforms as th_transforms
from torch import nn
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA
import cv2

import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.layers.activation import SiLU
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm

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


class CLIPViTEncoderOverloaded(nn.Module):

    """
    Vision transformer encoder for CLIP.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        patch_size (int): The dimension of each patch
        image_size(int): The size (width==height) of input image
        width (int): Dimensionality of the encoder layers and the pooler layer
        heads (int): Number of attention heads for each attention layer in the Transformer encoder
        layers (int): Number of hidden layers in the Transformer encoder

    Inputs:
        x (Tensor): image tensor with dimensions B x C(3) x image_size x image_size
    """

    def __init__(
        self,
        embedding_dim: int,
        patch_size: int,
        image_size: int,
        width: int,
        heads: int,
        layers: int,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.image_size = image_size

        scale = width**-0.5
        self.cls_token_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = Fp32LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dropout=0.0,
            activation=SiLU(),
            norm_first=True,
            dim_feedforward=4 * width,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.ln_post = Fp32LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:

        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(
                f"Expected input with width and height as {self.image_size}, found {x.size(2)} by {x.size(3)} "
            )
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels found {x.size(1)}")

        # B x C x image_size x image_size => B x width (out_channel) x patch_size x patch_size
        x = self.conv(x)

        # B x width x patch_size x patch_size => B x width x patch_size ** 2
        x = torch.flatten(x, start_dim=2)

        # B x width x patch_size ** 2 => B x patch_size ** 2 x width
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = self.encoder(x)

        # override for positional tokens
        # Take embedding of the cls token
        x_cls = self.ln_post(x[:, 0, :])
        x_cls = x_cls @ self.projection

        return self.ln_post(x[:, 1:, :])


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
        #scale = 14 ** -0.5
        #self.token_projection = nn.Parameter(scale * torch.randn(14, 14, 768))

    def forward(
        self,
        features_a: torch.Tensor,
    ):

        embeddings_a = self.encoder_a(features_a)
        #embeddings_a = embeddings_a @ self.token_projection
        embeddings_a = F.normalize(embeddings_a)
        return embeddings_a

    def transform_img(self, image):
        return self.torch_transform(image)


def clip_vit_b16(pretrained: bool = False) -> ClipOverloaded:
    vision_encoder = CLIPViTEncoderOverloaded(
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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_size = (224, 224)
    # model instantiation

    vl_model = clip_vit_b16(pretrained=True)
    vl_model = vl_model.to(device)

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

    embedding = vl_model(image_batch)

    img_emb = embedding.detach().cpu()
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