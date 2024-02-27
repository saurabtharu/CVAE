import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import cv2


import config


class NoisyDataset(Dataset):
    def __init__(self, dataset, noise_std):
        self.dataset = dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Add noise to the image
        noisy_image = image + torch.randn_like(image) * self.noise_std
        noisy_image = torch.clamp(
            noisy_image, 0, 1
        )  # Ensure pixel values are in [0, 1] range

        return noisy_image, label


class BlurryDataset(Dataset):
    def __init__(self, dataset, blur_radius):
        self.dataset = dataset
        self.blur_radius = blur_radius

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Convert PIL image to numpy array
        image_np = transforms.ToPILImage()(image)
        image_np = np.array(image_np)

        # Apply blur using cv2.blur
        blurry_image = cv2.blur(image_np, (self.blur_radius, self.blur_radius))

        # Convert numpy array back to tensor
        blurry_image = transforms.ToTensor()(blurry_image)

        return blurry_image, label


# Define transformation to be applied to the images
transform = transforms.Compose(
    [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # Resize to 256x256
        transforms.ToTensor(),  # Convert to tensor
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ]
)

# Create dataset using ImageFolder
celebface_dataset = datasets.ImageFolder(
    root=config.CELEBFACE_ROOT, transform=transform
)


# Split dataset into training and validation sets
train_size = int(0.8 * len(celebface_dataset))  # 80% for training
val_size = len(celebface_dataset) - train_size  # 20% for validation
# celeb_train_dataset, celeb_val_dataset = celebface_dataset[:train_size], celebface_dataset[train_size:val_size]
celeb_train_dataset, celeb_val_dataset = random_split(
    celebface_dataset, [train_size, val_size]
)

# Create DataLoader for training set
### Train dataloader
celeb_train_dataloader = DataLoader(
    dataset=celeb_train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
)

## Test dataloader
# Create DataLoader for validation set
celeb_val_dataloader = DataLoader(
    dataset=celeb_val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
)


def create_noisy_dataloader(dataset, noise_std, batch_size=config.BATCH_SIZE):
    # Create noisy dataset
    noisy_dataset = NoisyDataset(dataset, noise_std)

    # Create noisy data loader
    loader = DataLoader(
        dataset=noisy_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader


noise_std = 0.6
celeb_noisy_train_dataloader = create_noisy_dataloader(
    celeb_train_dataset, noise_std, config.BATCH_SIZE
)
celeb_noisy_val_dataloader = create_noisy_dataloader(
    celeb_val_dataset, noise_std, config.BATCH_SIZE
)


def create_blurry_dataloader(dataset, blur_radius, batch_size=config.BATCH_SIZE):
    # Create noisy dataset
    blurry_dataset = BlurryDataset(dataset=dataset, blur_radius=blur_radius)
    #     noisy_dataset = NoisyDataset(dataset, noise_std)

    # Create noisy data loader
    loader = DataLoader(
        dataset=blurry_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader


create_blurry_dataloader(
    dataset=celeb_val_dataloader, blur_radius=20, batch_size=config.BATCH_SIZE
)
