from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        raise NotImplementedError

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths
        valid = torch.ones((batch_size, 1), device=self.device, dtype=torch.float32) 
        fake = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32) 
        
        # TODO: Create noise and labels for generator input
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        labels = torch.randint(0, self.hparams.n_classes, (batch_size,), device=self.device)


        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator
            
            # TODO: Generate a batch of images
            generated_imgs = self(z, labels)

            # TODO: Calculate loss to measure generator's ability to fool the discriminator
            g_loss = self.adversarial_loss(self.discriminator(generated_imgs, labels), valid)
            log_dict["g_loss"] = g_loss
            return log_dict, g_loss
        
        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator

            # TODO: Generate a batch of images
            fake_imgs = self.generator(z, labels).detach()

            # TODO: Calculate loss for real images
            real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)

            # TODO: Calculate loss for fake images
            fake_loss = self.adversarial_loss(self.discriminator(fake_imgs, labels), fake)
            
            # TODO: Calculate total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            log_dict["d_loss"] = d_loss
            return log_dict, d_loss
        return log_dict, loss

    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch
        z = torch.randn(64, self.hparams.latent_dim, device=self.device) 
        labels = torch.randint(0, self.hparams.n_classes, (64,), device=self.device) 

        # TODO: Create fake images
        fake_images = self.generator(z, labels)
        fake_images = (fake_images + 1) / 2  
        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object
                logger.experiment.log({"gen_imgs": [wandb.Image(img) for img in fake_images]})
