import torch
import torch.optim as optim
import os

import config
import dataloader 
from CVAE import CVAE
import train


def main():
    # set random seeds 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    model = CVAE(img_size=config.IMAGE_SIZE).to(config.DEVICE)
    state_dict = torch.load('./models/model_1.pth')
    model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    ## training of model
    train.train_func(
        model=model, 
        epochs=config.EPOCHS,
        optimizer=optimizer, 
        train_dataloader=dataloader.celeb_train_dataloader, 
        noisy_train_dataloader=dataloader.celeb_noisy_train_dataloader
    )



    # Saving model
    model_state_dict = model.state_dict()
    save_path = './models/model_1_200e.pth'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model state dictionary to the specified path
    torch.save(model_state_dict, save_path)

if __name__ == '__main__':
    print(config.DEVICE)
    main()