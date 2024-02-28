import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


class CVAE_v1(nn.Module):
    def __init__(self, img_channels=3, z_dim=256, img_size=64):
        super(CVAE_v1, self).__init__()
        self.img_size = img_size

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )

        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )

        # Determine the size of the output of the last convolutional layer
        conv_out_size = self.calculate_conv_output_size(img_size)

        self.enc_fc_mu = nn.Linear(128 * conv_out_size * conv_out_size, z_dim)
        self.enc_fc_logvar = nn.Linear(128 * conv_out_size * conv_out_size, z_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(z_dim, 128 * conv_out_size * conv_out_size)
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=img_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def calculate_conv_output_size(self, img_size):
        # Calculate the output size after passing through the convolutional layers
        out_size = img_size
        for _ in range(3):  # 3 convolutional layers in encoder and decoder
            out_size = (
                out_size + 2 - 4
            ) // 2 + 1  # Assuming kernel_size=4, stride=2, padding=1
        return out_size

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(
            -1,
            128
            * self.calculate_conv_output_size(self.img_size)
            * self.calculate_conv_output_size(self.img_size),
        )  # Flatten
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(
            -1,
            128,
            self.calculate_conv_output_size(self.img_size),
            self.calculate_conv_output_size(self.img_size),
        )  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar


class CVAE_v2(nn.Module):
    def __init__(self, img_channels=3, z_dim=256, img_size=64):
        super(CVAE_v2, self).__init__()
        self.img_size = img_size

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )

        # Determine the size of the output of the last convolutional layer
        conv_out_size = self.calculate_conv_output_size(img_size)

        self.enc_fc_mu = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)
        self.enc_fc_logvar = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(z_dim, 256 * conv_out_size * conv_out_size)
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=img_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def calculate_conv_output_size(self, img_size):
        # Calculate the output size after passing through the convolutional layers
        out_size = img_size
        for _ in range(4):  # 4 convolutional layers in encoder
            out_size = (
                out_size + 2 - 4
            ) // 2 + 1  # Assuming kernel_size=4, stride=2, padding=1
        return out_size

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(
            -1,
            256
            * self.calculate_conv_output_size(self.img_size)
            * self.calculate_conv_output_size(self.img_size),
        )  # Flatten
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(
            -1,
            256,
            self.calculate_conv_output_size(self.img_size),
            self.calculate_conv_output_size(self.img_size),
        )  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar


class CVAE_v3(nn.Module):
    def __init__(self, img_channels=3, z_dim=256, img_size=64):
        super(CVAE_v3, self).__init__()
        self.img_size = img_size

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )

        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )

        self.enc_conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )

        self.enc_conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
        )

        # Determine the size of the output of the last convolutional layer
        conv_out_size = self.calculate_conv_output_size(img_size)

        self.enc_fc_mu = nn.Linear(512 * conv_out_size * conv_out_size, z_dim)
        self.enc_fc_logvar = nn.Linear(512 * conv_out_size * conv_out_size, z_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(z_dim, 512 * conv_out_size * conv_out_size)
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv5 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=img_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def calculate_conv_output_size(self, img_size):
        # Calculate the output size after passing through the convolutional layers
        out_size = img_size
        for _ in range(5):  # 5 convolutional layers in encoder and decoder
            out_size = (
                out_size + 2 - 4
            ) // 2 + 1  # Assuming kernel_size=4, stride=2, padding=1
        return out_size

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = F.relu(self.enc_conv5(x))
        x = x.view(
            -1,
            512
            * self.calculate_conv_output_size(self.img_size)
            * self.calculate_conv_output_size(self.img_size),
        )  # Flatten
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(
            -1,
            512,
            self.calculate_conv_output_size(self.img_size),
            self.calculate_conv_output_size(self.img_size),
        )  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        x = torch.sigmoid(self.dec_conv5(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar


class CVAE_v4(nn.Module):
    def __init__(self, img_channels=3, z_dim=256, img_size=64):
        super(CVAE_v4, self).__init__()
        self.img_size = img_size

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )

        # Determine the size of the output of the last convolutional layer
        conv_out_size = self.calculate_conv_output_size(img_size)

        self.enc_fc_mu = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)
        self.enc_fc_logvar = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(
            in_features=z_dim, out_features=256 * conv_out_size * conv_out_size
        )
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=img_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def calculate_conv_output_size(self, img_size):
        # Calculate the output size after passing through the convolutional layers
        out_size = img_size
        for _ in range(4):  # 4 convolutional layers in encoder
            out_size = (
                out_size + 2 - 4
            ) // 2 + 1  # Assuming kernel_size=4, stride=2, padding=1
        return out_size

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(
            -1,
            256
            * self.calculate_conv_output_size(self.img_size)
            * self.calculate_conv_output_size(self.img_size),
        )  # Flatten
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(
            -1,
            256,
            self.calculate_conv_output_size(self.img_size),
            self.calculate_conv_output_size(self.img_size),
        )  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar


class CVAE_dropout(nn.Module):
    def __init__(self, img_channels=3, z_dim=256, img_size=64, dropout_prob=0.5):
        super(CVAE_dropout, self).__init__()
        self.img_size = img_size
        self.dropout_prob = dropout_prob

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dropout1 = nn.Dropout(dropout_prob)

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dropout2 = nn.Dropout(dropout_prob)

        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dropout3 = nn.Dropout(dropout_prob)

        self.enc_conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )
        self.dropout4 = nn.Dropout(dropout_prob)

        # Determine the size of the output of the last convolutional layer
        conv_out_size = self.calculate_conv_output_size(img_size)

        self.enc_fc_mu = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)
        self.enc_fc_logvar = nn.Linear(256 * conv_out_size * conv_out_size, z_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(z_dim, 256 * conv_out_size * conv_out_size)
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dropout5 = nn.Dropout(dropout_prob)

        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dropout6 = nn.Dropout(dropout_prob)

        self.dec_conv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dropout7 = nn.Dropout(dropout_prob)

        self.dec_conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=img_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def calculate_conv_output_size(self, img_size):
        # Calculate the output size after passing through the convolutional layers
        out_size = img_size
        for _ in range(4):  # 4 convolutional layers in encoder
            out_size = (
                out_size + 2 - 4
            ) // 2 + 1  # Assuming kernel_size=4, stride=2, padding=1
        return out_size

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.enc_conv3(x))
        x = self.dropout3(x)
        x = F.relu(self.enc_conv4(x))
        x = self.dropout4(x)
        x = x.view(
            -1,
            256
            * self.calculate_conv_output_size(self.img_size)
            * self.calculate_conv_output_size(self.img_size),
        )  # Flatten
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(
            -1,
            256,
            self.calculate_conv_output_size(self.img_size),
            self.calculate_conv_output_size(self.img_size),
        )  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = self.dropout5(x)
        x = F.relu(self.dec_conv2(x))
        x = self.dropout6(x)
        x = F.relu(self.dec_conv3(x))
        x = self.dropout7(x)
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar


def loss_function(recon_img, input_img, mu, logvar, beta=1.0):
    #     print(f"input img size = {input_img.shape}")
    #     print(f"reconstructed img size = {recon_img.shape}")
    input_img = input_img.view(recon_img.size())
    #     print(f"after view input img size = {input_img.shape}")

    BCE = nn.functional.binary_cross_entropy(recon_img, input_img, reduction="sum")
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + beta * KLD
