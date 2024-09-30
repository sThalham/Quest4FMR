import torch
from PIL import Image
import torchvision.transforms as transforms

# Flava import
from torchmultimodal.models.flava.model import flava_model
#from torchmultimodal.models.flava.data.transforms import (
#    default_image_pretraining_transforms,
#    default_text_transform,
#)

from torchmultimodal.models.blip2.blip2 import BLIP2
#from multimodal.examples.flava.data import default_text_transform
#from flava.data import default_image_pretraining_transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flava
    #flava = flava_model(pretrained=True)
    # BLIP
    vl_model = BLIP2()

    vl_model = vl_model.to(device)
    #vl_model.eval()

    im_transform = transforms.ToTensor()

    text_embeds = []
    image_embeds = []

    img_path = "/home/stefan/datasets/hope_base/hope/onboarding_static/obj_000001_down/rgb/000000.jpg"
    image = []
    with Image.open(img_path) as im:
        width, height = im.size
        new_size = (width // 2, height // 2)
        left = (new_size[0] - 224) // 2
        top = (new_size[1] - 224) // 2
        right = left + 224
        bottom = top + 224
        im = im.resize(new_size)

        im = im.crop((left, top, right, bottom))

        tensor_image = im_transform(im)
        image_batch = torch.unsqueeze(tensor_image, 0)

        # The image is now a PyTorch tensor
    print(image_batch.shape)

        #_, text_emb = flava.encode_text(text.to(device), projection=True)
    _, image_emb = vl_model.encode_image(image_batch.to(device), projection=True)
    #text_emb = text_emb.detach().cpu()
    img_emb = image_emb.detach().cpu()

    print(img_emb.shape)

if __name__ == "__main__":
    main()