import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from config import DEVICE, IMAGE_SIZE
from CVAE import CVAE_v4, CVAE_dropout
from dataloader import celeb_val_dataloader, celeb_noisy_val_dataloader, transform


def load_model(model_path, model_arch):
    model = model_arch(img_size=IMAGE_SIZE).to(DEVICE)

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))

    if next(iter(state_dict.keys())).startswith("module"):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    return model


model = load_model("./models/best_model.pth", CVAE_v4)


noisy_image_path = "./test_image/noisy_image.jpg"
original_image_path = "./test_image/original_image.jpg"


noisy_image = Image.open(noisy_image_path)
original_image = Image.open(original_image_path)


# Convert images to tensors and move to device
noisy_image = transform(noisy_image).unsqueeze(0).to(DEVICE)
original_image = transform(original_image).unsqueeze(0).to(DEVICE)
model.eval()
with torch.inference_mode():
    reconstructed_image, _, _ = model(noisy_image.squeeze(0).to(DEVICE))

# index = 32
# original_image = celeb_val_dataloader.dataset[index][0]
# noisy_image = celeb_noisy_val_dataloader.dataset[index][0]
#
# model.eval()
# with torch.inference_mode():
#     reconstructed_image, _, _ = model(noisy_image.to(DEVICE))
#

plt.subplot(1, 3, 1)
plt.imshow(original_image.squeeze().permute(1, 2, 0))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image.squeeze().permute(1, 2, 0))
plt.title("Nosiy Image")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image.squeeze().cpu().permute(1, 2, 0))
plt.title("Reconstructed")
plt.axis("off")

plt., CVAE_dropout
