import torch
from torchvision import transforms
from PIL import Image
import os


def add_gaussian_noise(image_path, noise_std=0.1):
    # Load image
    image = Image.open(image_path)

    # Convert image to tensor
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Add Gaussian noise
    noisy_image = image + torch.randn_like(image) * noise_std
    noisy_image = torch.clamp(
        noisy_image, 0, 1
    )  # Ensure pixel values are in [0, 1] range

    # Convert noisy image tensor to PIL image
    noisy_image = transforms.ToPILImage()(noisy_image.squeeze(0))

    # Get the directory of the image
    directory = os.path.dirname(image_path)

    # Save the noisy image
    noisy_image_path = os.path.join(directory, "noisy_" + os.path.basename(image_path))
    noisy_image.save(noisy_image_path)

    return noisy_image_path


# Example usage
image_path = "./test_image/nihang.jpg"
noisy_image_path = add_gaussian_noise(image_path, noise_std=0.6)
print("Noisy image saved at:", noisy_image_path)
