import torch
from PIL import Image
import torchvision.transforms as transforms

from octo.octo.model.octo_model import OctoModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "/home/stefan/Data/MMFMR/octo/octo_small"
    vl_model = OctoModel.load_pretrained(model_path)

    vl_model = vl_model.to(device)
    #vl_model.eval()

    im_transform = transforms.ToTensor()

    img_path = "/home/stefan/Quest4FMR/images/query_0.png"
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