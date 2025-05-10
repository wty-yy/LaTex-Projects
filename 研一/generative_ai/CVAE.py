import torch
import torch.nn as nn
import torch.nn.functional as F


# Variational Autoencoder
class CVAE(nn.Module):

    def __init__(self, latent_size=64):
        super(CVAE, self).__init__()
        self.labels = 10  # Number of labels
        self.latent_size = latent_size  # Dimension of latent space
        # Encoder layers: input is flattened image and label
        self.fc1 = nn.Linear(28 * 28 + self.labels, 512)  # Encoder input layer
        self.fc2 = nn.Linear(512, latent_size)              # Mean of latent space
        self.fc3 = nn.Linear(512, latent_size)              # Log variance of latent space

        # Decoder layers: input is concatenation of latent variable and label
        self.fc4 = nn.Linear(latent_size + self.labels, 512)  # Decoder input layer
        self.fc5 = nn.Linear(512, 28 * 28)                    # Decoder output layer

    # Encoder part, input and output are consistent with ConvCVAE
    def encode(self, x, y):
        # x: (batch, 1, 28, 28), y: (batch, 10)
        x_flat = x.view(x.size(0), -1)  # Flatten the image
        x_concat = torch.cat([x_flat, y], dim=1)
        hidden = F.relu(self.fc1(x_concat))
        mu = self.fc2(hidden)
        log_var = self.fc3(hidden)
        return mu, log_var

    # Reparameterization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decoder part, input and output are consistent with ConvCVAE
    def decode(self, z, y):
        # z: (batch, latent_size), y: (batch, 10)
        z_concat = torch.cat([z, y], dim=1)
        hidden = F.relu(self.fc4(z_concat))
        recon_x = torch.sigmoid(self.fc5(hidden))
        return recon_x

    # Forward pass, interface identical to ConvCVAE
    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var


class ConvCVAE(nn.Module):

    def __init__(self, latent_size=64):
        super(ConvCVAE, self).__init__()
        self.labels = 10
        # Encoder: image channels (1) concat one-hot label (10) --> 11 channels
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU())
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_size)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_size)

        # Decoder: concatenated latent vector with label (latent_size+10) -> feature map
        self.fc_decode = nn.Linear(latent_size + 10, 64 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid())

    def encode(self, x, y):
        # x shape: (batch, 1, 28, 28), y shape: (batch, 10)
        # Expand y spatially and concatenate with x along channel dimension.
        y_expanded = y.view(y.size(0), y.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x_cat = torch.cat([x, y_expanded], dim=1)  # Now: (batch, 11, 28, 28)
        conv_out = self.encoder_conv(x_cat)  # (batch, 64, 7, 7)
        conv_out = conv_out.view(conv_out.size(0), -1)
        mu = self.fc_mu(conv_out)
        logvar = self.fc_logvar(conv_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # Concatenate latent vector with label.
        z_cat = torch.cat([z, y], dim=1)
        fc_out = self.fc_decode(z_cat)
        fc_out = fc_out.view(-1, 64, 7, 7)
        recon_x = self.decoder_deconv(fc_out)
        return recon_x

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar
