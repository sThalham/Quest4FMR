import torch
from PIL import Image
import math
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Flava import
from torchmultimodal.models.flava.model import FLAVAModel, flava_multimodal_encoder
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Tuple, Union
from torchmultimodal.models.flava.image_encoder import flava_image_encoder
from torchmultimodal.models.flava.text_encoder import flava_text_encoder
from multimodal.multimodal.examples.mugen.data.bert_text_transform import BertTextTransform
from torchmultimodal.utils.common import load_module_from_url, ModelOutput
from sklearn.decomposition import PCA
from torchvision import transforms as th_transforms

CKPT_KEY = "flava_full"
FLAVA_MODEL_MAPPING = {
    CKPT_KEY: "https://download.pytorch.org/models/multimodal/flava/flava_model_unified_text_encoder.pt",
}


def flava_model(
    # Image encoder specific parameters
    image_hidden_size: int = 768,
    image_num_attention_heads: int = 12,
    image_num_hidden_layers: int = 12,
    image_dropout: float = 0.0,
    image_intermediate_size: int = 3072,
    image_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    image_layer_norm_eps: float = 1e-12,
    use_image_masking: bool = True,
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
    # Text encoder specific parameters
    text_hidden_size: int = 768,
    text_num_attention_heads: int = 12,
    text_num_hidden_layers: int = 12,
    text_dropout: float = 0.0,
    text_intermediate_size: int = 3072,
    text_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    text_layer_norm_eps: float = 1e-12,
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
    # Multimodal encoder specific parameters
    multimodal_hidden_size: int = 768,
    multimodal_num_attention_heads: int = 12,
    multimodal_num_hidden_layers: int = 6,
    multimodal_dropout: float = 0.0,
    multimodal_intermediate_size: int = 3072,
    multimodal_intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    multimodal_layer_norm_eps: float = 1e-12,
    # projection
    text_and_image_proj_size: int = 768,
    pretrained: bool = True,
    **kwargs: Any,
) -> FLAVAModel:
    image_encoder = flava_image_encoder(
        hidden_size=image_hidden_size,
        num_attention_heads=image_num_attention_heads,
        num_hidden_layers=image_num_hidden_layers,
        use_image_masking=use_image_masking,
        dropout=image_dropout,
        intermediate_size=image_intermediate_size,
        intermediate_activation=image_intermediate_activation,
        layer_norm_eps=image_layer_norm_eps,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
    )
    text_encoder = flava_text_encoder(
        hidden_size=text_hidden_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
        dropout=text_dropout,
        intermediate_size=text_intermediate_size,
        intermediate_activation=text_intermediate_activation,
        layer_norm_eps=text_layer_norm_eps,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
    )
    mm_encoder = flava_multimodal_encoder(
        hidden_size=multimodal_hidden_size,
        num_attention_heads=multimodal_num_attention_heads,
        num_hidden_layers=multimodal_num_hidden_layers,
        dropout=multimodal_dropout,
        intermediate_size=multimodal_intermediate_size,
        intermediate_activation=multimodal_intermediate_activation,
        layer_norm_eps=multimodal_layer_norm_eps,
    )

    image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
    text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)

    image_projection = nn.Linear(image_hidden_size, text_and_image_proj_size)
    text_projection = nn.Linear(text_hidden_size, text_and_image_proj_size)

    flava = FLAVAModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        mm_encoder=mm_encoder,
        image_to_mm_projection=image_to_mm_projection,
        text_to_mm_projection=text_to_mm_projection,
        text_projection=text_projection,
        image_projection=image_projection,
    )

    if pretrained:
        load_module_from_url(flava, FLAVA_MODEL_MAPPING[CKPT_KEY])

    return flava


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im_size = (224, 224)
    vl_model = flava_model(pretrained=True)
    vl_model = vl_model.to(device)
    #vl_model.eval()

    text_transform = BertTextTransform()
    torch_transform = th_transforms.Compose([
        th_transforms.Resize(im_size),
        th_transforms.ToTensor(),
        th_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_path = "/home/stefan/Quest4FMR/images/query_0.png"
    with Image.open(img_path) as im:
        im_og = (np.array(im) * (1 / 255))
        o_height, o_width, _ = im_og.shape
        print(o_height, o_width)
        image_batch = torch_transform(im)
        image_batch = torch.unsqueeze(image_batch, 0).to(device)

    text = text_transform('cat')

    text_emb = vl_model.encode_text(text.to(device), projection=False)
    img_emb = vl_model.encode_image(image_batch, projection=False)
    img_emb = img_emb.last_hidden_state[:, 1:, :]
    print('text emb: ', text_emb.shape)

    b, tokens_text, feat_text = text_emb.shape
    text_emb = img_emb.detach().cpu()
    b, tokens, feat = img_emb.shape
    img_emb = img_emb.detach().cpu()

    # stupid projection to image space
    pca = PCA(n_components=3, svd_solver='full')
    img_viz = img_emb[0, ...]
    img_viz = pca.fit_transform(img_viz)
    #img_viz = pca.transform(img_viz)
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