import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)
        ch_att = torch.sigmoid(
            self.mlp(avg_pool.flatten(1))[:, :, None, None]
            + self.mlp(max_pool.flatten(1))[:, :, None, None]
        )
        x = x * ch_att
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * sp_att


class AE_byhand(nn.Module):
    def __init__(
        self,
        channels,
        dim_input,
        kernel_size,
        Nfc,
        latent_dim,
        max_pooling_kernel=2,
        in_channels=1,
    ):
        super().__init__()
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel
        self.in_channels = in_channels

        reduced_size = self.dim_input // (self.max_pooling_kernel ** 2)
        flattened_dim = reduced_size * reduced_size

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.MaxPool2d(self.max_pooling_kernel),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.MaxPool2d(self.max_pooling_kernel),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(in_channels=self.channels[2], out_channels=1, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, self.Nfc),
            nn.ELU(),
            nn.Linear(self.Nfc, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.Nfc),
            nn.ELU(),
            nn.Linear(self.Nfc, flattened_dim),
            nn.ELU(),
            nn.Unflatten(1, (1, reduced_size, reduced_size)),
            nn.Conv2d(in_channels=1, out_channels=self.channels[2], kernel_size=1),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Upsample(scale_factor=self.max_pooling_kernel, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=self.channels[2],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Upsample(scale_factor=self.max_pooling_kernel, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=self.channels[1],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=1,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class VAE(AE_byhand):
    def __init__(
        self,
        channels,
        dim_input,
        kernel_size,
        Nfc,
        latent_dim,
        max_pooling_kernel=2,
        in_channels=1,
    ):
        super().__init__(
            channels,
            dim_input,
            kernel_size,
            Nfc,
            latent_dim,
            max_pooling_kernel,
            in_channels,
        )

        reduced_size = self.dim_input // (self.max_pooling_kernel ** 2)
        flattened_dim = reduced_size * reduced_size

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.MaxPool2d(self.max_pooling_kernel),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.MaxPool2d(self.max_pooling_kernel),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(in_channels=self.channels[2], out_channels=1, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, self.Nfc),
            nn.ELU(),
        )

        self.fc_mu = nn.Linear(self.Nfc, self.latent_dim)
        self.fc_logvar = nn.Linear(self.Nfc, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.Nfc),
            nn.ELU(),
            nn.Linear(self.Nfc, flattened_dim),
            nn.ELU(),
            nn.Unflatten(1, (1, reduced_size, reduced_size)),
            nn.Conv2d(in_channels=1, out_channels=self.channels[2], kernel_size=1),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            CBAM(self.channels[2]),
            nn.Upsample(scale_factor=self.max_pooling_kernel, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=self.channels[2],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Conv2d(
                in_channels=self.channels[1],
                out_channels=self.channels[1],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            CBAM(self.channels[1]),
            nn.Upsample(scale_factor=self.max_pooling_kernel, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=self.channels[1],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=self.channels[0],
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            CBAM(self.channels[0]),
            nn.Conv2d(
                in_channels=self.channels[0],
                out_channels=1,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


class AEModule(LightningModule):
    def __init__(
        self,
        channels,
        dim_input,
        kernel_size,
        Nfc,
        latent_dim,
        max_pooling_kernel,
        lr,
        threshold,
        wBCE,
        in_channels=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel
        self.lr = lr
        self.threshold = threshold
        self.wBCE = wBCE

        self.model = AE_byhand(
            channels=self.channels,
            dim_input=self.dim_input,
            kernel_size=self.kernel_size,
            Nfc=self.Nfc,
            latent_dim=self.latent_dim,
            max_pooling_kernel=self.max_pooling_kernel,
            in_channels=in_channels,
        )

        self.accuracy = Accuracy(task="binary", num_classes=2, average="macro")
        self.test_precision = Precision(task="binary", num_classes=2, average="macro")
        self.recall = Recall(task="binary", num_classes=2, average="macro")
        self.F1Score = F1Score(task="binary", num_classes=2, average="macro")

        self.errors_list = []
        self.clean_errors = []
        self.anomalous_errors = []
        self.z_values_anomalous = []
        self.z_values_clean = []

    def weighted_BCELoss(self, pred, gt, batch_mean=False):
        # ensure ground truth has same channel count as prediction
        if gt.shape[1] != pred.shape[1]:
            gt = gt[:, : pred.shape[1], :, :]
        weight_map = torch.ones_like(pred)
        weight_map[gt != 0] = self.wBCE
        restore_error = -torch.mean(
            weight_map
            * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12)),
            dim=(1, 2, 3),
        )
        if batch_mean:
            return restore_error
        return torch.mean(restore_error)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        encoded, decoded = self(batch)
        loss = self.weighted_BCELoss(decoded, batch)
        self.log("Train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        _, decoded = self(batch)
        loss = self.weighted_BCELoss(decoded, batch)
        errors = self.weighted_BCELoss(decoded, batch, batch_mean=True)
        self.errors_list.append(errors.detach().cpu())
        self.log("Val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.errors_list:
            return
        errors = torch.cat(self.errors_list, dim=0)
        self.errors_list.clear()
        self.threshold = torch.quantile(errors, 0.95).item()
        self.log("Val/threshold", self.threshold, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        if image.dim() == 3:
            image = image.unsqueeze(1)
        encoded, decoded = self(image)
        pred = self.weighted_BCELoss(decoded, image, batch_mean=True)
        self.clean_errors.append(pred[label == 0].detach().cpu())
        self.anomalous_errors.append(pred[label == 1].detach().cpu())
        self.z_values_clean.append(encoded[label == 0].detach().cpu())
        self.z_values_anomalous.append(encoded[label == 1].detach().cpu())

        pred = (pred > self.threshold).long()
        self.accuracy(pred, label)
        self.test_precision(pred, label)
        self.recall(pred, label)
        self.F1Score(pred, label)

        self.log("Test/accuracy", self.accuracy, on_epoch=True)
        self.log("Test/precision", self.test_precision, on_epoch=True)
        self.log("Test/recall", self.recall, on_epoch=True)
        self.log("Test/F1Score", self.F1Score, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=2, factor=0.1
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "Val/loss"}}


class VAEModule(LightningModule):
    def __init__(
        self,
        channels,
        dim_input,
        kernel_size,
        Nfc,
        latent_dim,
        max_pooling_kernel,
        lr,
        threshold,
        wBCE,
        beta_max,
        kl_warmup_epochs,
        in_channels=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel
        self.lr = lr
        self.threshold = threshold
        self.wBCE = wBCE
        self.beta_max = beta_max
        self.kl_warmup_epochs = max(1, kl_warmup_epochs)
        self.current_beta = 0.0

        self.model = VAE(
            channels=self.channels,
            dim_input=self.dim_input,
            kernel_size=self.kernel_size,
            Nfc=self.Nfc,
            latent_dim=self.latent_dim,
            max_pooling_kernel=self.max_pooling_kernel,
            in_channels=in_channels,
        )

        self.accuracy = Accuracy(task="binary", num_classes=2, average="macro")
        self.test_precision = Precision(task="binary", num_classes=2, average="macro")
        self.recall = Recall(task="binary", num_classes=2, average="macro")
        self.F1Score = F1Score(task="binary", num_classes=2, average="macro")

        self.errors_list = []
        self.clean_errors = []
        self.anomalous_errors = []
        self.z_values_anomalous = []
        self.z_values_clean = []

    def on_train_epoch_start(self):
        ramp = min(1.0, (self.current_epoch + 1) / self.kl_warmup_epochs)
        self.current_beta = self.beta_max * ramp

    def weighted_BCELoss(self, pred, gt, batch_mean=False):
        if gt.shape[1] != pred.shape[1]:
            gt = gt[:, : pred.shape[1], :, :]
        weight_map = torch.ones_like(pred)
        weight_map[gt != 0] = self.wBCE
        restore_error = -torch.mean(
            weight_map
            * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12)),
            dim=(1, 2, 3),
        )
        if batch_mean:
            return restore_error
        return torch.mean(restore_error)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        decoded, mu, logvar = self(batch)
        recon_loss = self.weighted_BCELoss(decoded, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        final_loss = recon_loss + self.current_beta * kl_loss

        self.log("Train/loss", final_loss, prog_bar=True)
        self.log("Train/rec_err", recon_loss)
        self.log("Train/kl", kl_loss)
        self.log("Train/beta", self.current_beta, prog_bar=True)
        return final_loss

    def validation_step(self, batch, batch_idx):
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        decoded, mu, logvar = self(batch)
        recon_loss = self.weighted_BCELoss(decoded, batch)
        kl_vec = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss = recon_loss + self.current_beta * kl_vec.mean()
        errors = self.weighted_BCELoss(decoded, batch, batch_mean=True) + self.current_beta * kl_vec
        self.errors_list.append(errors.detach().cpu())
        self.log("Val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.errors_list:
            return
        errors = torch.cat(self.errors_list, dim=0)
        self.errors_list.clear()
        self.threshold = torch.quantile(errors, 0.95).item()
        self.log("Val/threshold", self.threshold, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        if image.dim() == 3:
            image = image.unsqueeze(1)
        decoded, mu, logvar = self(image)
        recon_err = self.weighted_BCELoss(decoded, image, batch_mean=True)
        kl_vec = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        errors = recon_err + self.current_beta * kl_vec

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        self.z_values_anomalous.append(z[label == 1].detach().cpu())
        self.z_values_clean.append(z[label == 0].detach().cpu())
        self.clean_errors.append(errors[label == 0].detach().cpu())
        self.anomalous_errors.append(errors[label == 1].detach().cpu())

        pred = (errors > self.threshold).long()
        self.accuracy(pred, label)
        self.test_precision(pred, label)
        self.recall(pred, label)
        self.F1Score(pred, label)

        self.log("Test/accuracy", self.accuracy, on_epoch=True)
        self.log("Test/precision", self.test_precision, on_epoch=True)
        self.log("Test/recall", self.recall, on_epoch=True)
        self.log("Test/F1Score", self.F1Score, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=2, factor=0.1
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "Val/loss"}}


if __name__ == "__main__":
    dim_input = 256
    channels = [16, 32, 64]
    kernel_size = 5
    Nfc = 128
    latent_dim = 20
    max_pooling_kernel = 2

    ae_module = AEModule(
        channels=channels,
        dim_input=dim_input,
        kernel_size=kernel_size,
        Nfc=Nfc,
        latent_dim=latent_dim,
        max_pooling_kernel=max_pooling_kernel,
        lr=0.001,
        threshold=0.5,
        wBCE=1,
        in_channels=3,
    )

    batch_size = 4
    test_images = torch.rand(batch_size, 3, dim_input, dim_input)
    encoded, reconstructed = ae_module(test_images)
    print("AE encoded:", encoded.shape, "decoded:", reconstructed.shape)

    vae_module = VAEModule(
        channels=channels,
        dim_input=dim_input,
        kernel_size=kernel_size,
        Nfc=Nfc,
        latent_dim=latent_dim,
        max_pooling_kernel=max_pooling_kernel,
        lr=0.001,
        threshold=0.5,
        wBCE=1,
        beta_max=0.1,
        kl_warmup_epochs=5,
        in_channels=3,
    )

    decoded, mu, logvar = vae_module(test_images)
    print("VAE decoded:", decoded.shape, "mu:", mu.shape, "logvar:", logvar.shape)

    summary(vae_module, (3, dim_input, dim_input), device="cpu")
