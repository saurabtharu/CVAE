import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from config import DEVICE
from CVAE import loss_function
import os


def train_func(model, epochs, optimizer, train_dataloader, noisy_train_dataloader):
    loss_history = []
    num = 1
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_loss = 0.0
        for (clean_img, _), (noisy_img, _) in zip(
            train_dataloader, noisy_train_dataloader
        ):
            clean_img = clean_img.to(DEVICE)
            noisy_img = noisy_img.to(DEVICE)

            recon_img, mu, logvar = model(noisy_img)

            if num % 100 == 0:
                print(f"{num}, ", end="")
            num += 1

            # calculate the loss
            loss = loss_function(
                recon_img=recon_img, input_img=clean_img, mu=mu, logvar=logvar
            )

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
        num = 1
        print()
        average_loss = total_loss / len(train_dataloader.dataset)
        loss_history.append(average_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")

    return loss_history


def train_step(
    model: torch.nn.Module,
    clean_train_dataloader: DataLoader,
    noisy_train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn=loss_function,
):
    loss_history = []
    train_loss = 0
    num = 1
    average_loss = 0.0

    # 1. model
    model.train()
    total_loss = 0.0
    for (clean_img, _), (noisy_img, _) in zip(
        clean_train_dataloader, noisy_train_dataloader
    ):
        clean_img, noisy_img = clean_img.to(DEVICE), noisy_img.to(DEVICE)

        # 2. forward pass
        recon_img, mu, logVar = model(noisy_img)

        if num % 100 == 0:
            print(f"{num}, ", end="")
        num += 1

        # 3. calculate the loss
        loss = loss_fn(recon_img=recon_img, input_img=clean_img, mu=mu, logvar=logVar)

        # 4. optimizer zero grad
        optimizer.zero_grad()

        # 5. loss backward
        loss.backward()

        ### accumulating the training loss
        train_loss += loss.item()

        # 6. optimizer step
        optimizer.step()

        average_loss = total_loss / len(clean_train_dataloader.dataset)
    train_loss = train_loss / len(clean_train_dataloader.dataset)
    loss_history.append(average_loss)

    return train_loss


def test_step(
    model: torch.nn.Module,
    clean_test_dataloader: torch.utils.data.DataLoader,
    noisy_test_dataloader: torch.utils.data.DataLoader,
    loss_fn,
    device: torch.device,
):
    ### TESTING
    test_loss = 0

    # Put the model into eval mode
    model.eval()

    # turn on inference mode
    with torch.inference_mode():
        #     for X, y in data_loader:
        for (clean_test_img, _), (noisy_test_img, _) in zip(
            clean_test_dataloader, noisy_test_dataloader
        ):
            # put the data X and y into  device
            clean_test_img, noisy_test_img = clean_test_img.to(
                device
            ), noisy_test_img.to(device)

            # 1. forward pass (outputs raw logits )
            reconstructed_test_img, mu, logVar = model(noisy_test_img)

            # 2. calculate  the loss (accumulatively)
            loss = loss_fn(
                recon_img=reconstructed_test_img,
                input_img=clean_test_img,
                mu=mu,
                logvar=logVar,
            )
            #         loss = loss_fn(reconstructed_test_img, clean_test_img)
            test_loss += loss.item()

            # 3. calculate accuracy
        #         test_pred_labels = test_pred_logits.argmax(dim=1)
        #         test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)      # go from raw logits -> prediction labels

        # calculate the test loss and average per batc/h
        test_loss /= len(clean_test_dataloader.dataset)

        # calculate the test accuracy average per batch
        #     test_acc /= len(data_loader)

        # print what's happening
        # print(f"\nTest loss: {train_loss:.4f}, Test accuracy: {test_acc:.4f}% \n")
        return test_loss


def train(
    model: torch.nn.Module,
    clean_train_dataloader: torch.utils.data.DataLoader,
    noisy_train_dataloader: torch.utils.data.DataLoader,
    clean_val_dataloader: torch.utils.data.DataLoader,
    noisy_val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    # scheduler=None,
    device: torch.device = torch.device(DEVICE),
):
    results = {
        "train_loss": [],
        # "train_acc": [],
        "test_loss": [],
        # "test_acc": []
    }
    best_val_loss = float("inf")  # Initialize with a very high value
    consecutive_same_loss_count = 0
    prev_loss = None

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            clean_train_dataloader=clean_train_dataloader,
            noisy_train_dataloader=noisy_train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        if (epoch + 1) % 20 == 0 and epoch != 0:
            model_state_dict = model.state_dict()
            save_path = (
                f"./models/{model.__class__.__name__}/model_{epoch}epoch_trained.pth"
            )

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the model state dictionary to the specified path
            torch.save(model_state_dict, save_path)

        test_loss = test_step(
            model=model,
            clean_test_dataloader=clean_val_dataloader,
            noisy_test_dataloader=noisy_val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        # Update learning rate
        #         scheduler.step()

        # Save the model if the validation loss is improved
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            model_state_dict = model.state_dict()
            save_path = f"./models/{model.__class__.__name__}/best_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model_state_dict, save_path)

        print(
            f"Epoch: {epoch + 1} | Train loss : {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )
        results["train_loss"].append(train_loss)
        #     results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        #     results["test_acc"].append(test_acc)

        # Check if the current epoch has the same loss as the previous epoch
        if prev_loss is not None and train_loss == prev_loss:
            consecutive_same_loss_count += 1
        else:
            consecutive_same_loss_count = 0

        prev_loss = train_loss

        # If there are 4 consecutive epochs with the same loss, break out of the loop
        if consecutive_same_loss_count == 4:
            print("Training halted due to 4 consecutive epochs with the same loss.")
            break

    return results


def train_model(
    model,
    clean_train_dataloader,
    noisy_train_dataloader,
    clean_val_dataloader,
    noisy_val_dataloader,
    epochs,
    optimizer,
    state_dict_file=None,
):

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if state_dict_file:
        print(f"{state_dict_file} present!!!")
        state_dict = torch.load(state_dict_file)
        # If the model was trained using DataParallel, you need to remove the 'module.' prefix
        # from all keys in the state_dict
        if next(iter(state_dict.keys())).startswith("module"):
            state_dict = {key[7:]: value for key, value in state_dict.items()}

            # Load the state_dict into the model
            model.load_state_dict(state_dict)

    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=PATIENCE, verbose=True
    # )

    model_loss_results = train(
        model=model,
        clean_train_dataloader=clean_train_dataloader,
        noisy_train_dataloader=noisy_train_dataloader,
        clean_val_dataloader=clean_val_dataloader,
        noisy_val_dataloader=noisy_val_dataloader,
        optimizer=optimizer,
        # scheduler=scheduler,
        loss_fn=loss_function,
        epochs=epochs,
        device=torch.device(DEVICE),
    )
    return model_loss_results
