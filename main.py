import torch
import torch.optim as optim
import os
import json

import config
import dataloader
from CVAE import CVAE
import train
from generate_loss import plot_loss

"""

model_cvae_v2 = cvae_v2(img_size=512).to(device)
optimizer = torch.optim.adam(model_cvae_v2.parameters(), lr=lr)          # lr = alpha {learning rate}

# def train_model(model, epochs, optimizer, state_dict_file=none):
model_cvae_v2_result = train_model(model=model_cvae_v2, 
                                   epochs=epochs, 
                                   optimizer=optimizer, 
#                                    state_dict_file="./models/cvae_v1/model_6epoch_trained.pth"
                                  )


"""


def main():
    # set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = CVAE(img_size=config.IMAGE_SIZE).to(config.DEVICE)
    state_dict = torch.load(
        "./models/model_1_200e.pth", map_location=torch.device(config.DEVICE)
    )
    model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # def train_model(model, epochs, optimizer, state_dict_file=none):
    model_cvae_v2_result = train.train_model(
        model=model,
        clean_train_dataloader=dataloader.celeb_train_dataloader,
        noisy_train_dataloader=dataloader.celeb_noisy_train_dataloader,
        clean_val_dataloader=dataloader.celeb_val_dataloader,
        noisy_val_dataloader=dataloader.celeb_noisy_val_dataloader,
        epochs=config.EPOCHS,
        optimizer=optimizer,
        #                                    state_dict_file="./models/cvae_v1/model_6epoch_trained.pth"
    )

    print(json.dumps(model_cvae_v2_result, indent=4))
    with open("./loss.json", "w") as file:
        json.dump(model_cvae_v2_result, file)

    # Plotting the loss values
    plot_loss(model_cvae_v2_result, config.EPOCHS)

    # Saving model
    model_state_dict = model.state_dict()
    save_path = "./models/model_1_200e.pth"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model state dictionary to the specified path
    torch.save(model_state_dict, save_path)


if __name__ == "__main__":
    print(config.DEVICE)
    main()
