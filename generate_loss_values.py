import matplotlib.pyplot as plt
import loss_values

# import json


def plot_loss(data, total_epochs):  # Generate epochs list using total_epochs
    epochs = range(1, total_epochs + 1)
    train_loss = data["train_loss"]
    test_loss = data["test_loss"]
    # Plot train_loss and test_loss
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    # Add legend
    plt.legend()
    # Show plot
    plt.show()


merged_dict = {
    "train_loss": loss_values.dropout_loss_1["train_loss"]
    + loss_values.dropout_loss_2["train_loss"],
    "test_loss": loss_values.dropout_loss_1["test_loss"]
    + loss_values.dropout_loss_2["test_loss"],
}


# main_data = loss_values.dropout_loss_2
EPOCH = len(merged_dict["train_loss"])
plot_loss(merged_dict, EPOCH)
