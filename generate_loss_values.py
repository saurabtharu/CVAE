import matplotlib.pyplot as plt
import loss_values


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


main_data = loss_values.dropout_loss_1
EPOCH = len(main_data["train_loss"])
plot_loss(main_data, EPOCH)
