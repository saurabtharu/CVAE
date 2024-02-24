from config import DEVICE, IMAGE_SIZE
from CVAE import CVAE
import torch
from dataloader import celeb_val_dataloader, celeb_noisy_val_dataloader
import matplotlib.pyplot as plt


# Initialize your model architecture
model = CVAE(img_size=IMAGE_SIZE).to(DEVICE)

# Load the state_dict
state_dict = torch.load("./models/model_1.pth")

# If the model was trained using DataParallel, you need to remove the 'module.' prefix
# from all keys in the state_dict
if next(iter(state_dict.keys())).startswith("module"):
    state_dict = {key[7:]: value for key, value in state_dict.items()}

# Load the state_dict into the model
model.load_state_dict(state_dict)

index = 20

original_image = celeb_val_dataloader.dataset[index][0]

nosiy_image = celeb_noisy_val_dataloader.dataset[index][0]
model.eval()
with torch.inference_mode():
    reconstructed_image, _, _ = model(nosiy_image.unsqueeze(0).to(DEVICE))


plt.subplot(1, 3, 1)
plt.imshow(original_image.permute(1, 2, 0))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(nosiy_image.permute(1, 2, 0))
plt.title("Nosiy Image")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image.squeeze().cpu().permute(1, 2, 0))
plt.title("Reconstructed")
plt.axis("off")

plt.show()
