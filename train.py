from tqdm.auto import tqdm
from config import DEVICE
from CVAE import loss_function

def train_func(model, epochs, optimizer, train_dataloader, noisy_train_dataloader):
    loss_history = []
    num = 1
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_loss = 0.0
        for (clean_img, _), (noisy_img, _) in zip(train_dataloader, noisy_train_dataloader):
            clean_img = clean_img.to(DEVICE)
            noisy_img = noisy_img.to(DEVICE)

            recon_img, mu, logvar = model(noisy_img)
            
            if num % 100 == 0:
                print(f"{num}, ", end="")
            num += 1

            # calculate the loss
            loss = loss_function(recon_img=recon_img, input_img=clean_img, mu=mu, logvar=logvar)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
        num = 1
        print()
        average_loss = total_loss / len(train_dataloader.dataset)
        loss_history.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

    return loss_history
