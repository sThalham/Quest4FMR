import torch
from PIL import Image
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform

# Define helper function for zero-shot prediction
def predict(zero_shot_model, image, labels):
  zero_shot_model.eval()
  with torch.no_grad():
      image = image_transform(img)["image"].unsqueeze(0)
      texts = text_transform(labels)
      _, image_features = zero_shot_model.encode_image(image, projection=True)
      _, text_features = zero_shot_model.encode_text(texts, projection=True)
      scores = image_features @ text_features.t()
      probs = torch.nn.Softmax(dim=-1)(scores)
      label = labels[torch.argmax(probs)]
      print(
          "Label probabilities: ",
          {labels[i]: probs[:, i] for i in range(len(labels))},
      )
      print(f"Predicted label: {label}")


image_transform = FLAVAImageTransform(is_train=False)
text_transform = BertTextTransform()
zero_shot_model = flava_model(pretrained=True)
img = Image.open("my_image.jpg")  # point to your own image
predict(zero_shot_model, img, ["dog", "cat", "house"])

# Example output:
# Label probabilities:  {'dog': tensor([0.80590]), 'cat': tensor([0.0971]), 'house': tensor([0.0970])}
# Predicted label: dog