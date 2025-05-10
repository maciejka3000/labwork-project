import math
import time
from os import mkdir

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow.raft import ResidualBlock
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import shutil
from src.image_preprocess import ImageProcess
import cv2
import yaml
import glob
import random


def check_and_make(modelpath):
    if not os.path.exists(modelpath):
        print('creating {} folder'.format(modelpath))
        mkdir(modelpath)
    else:
        print('{} folder detected, removing'.format(modelpath))
        shutil.rmtree(modelpath)
        mkdir(modelpath)

def unnormalize(img_tensor):
    img_tensor_un = img_tensor.clone()
    img_tensor_un = img_tensor_un.clamp(0, 1)
    return img_tensor_un


def show_pair(inputs, targets, num_images = 1):
    inputs = inputs.cpu()
    targets = targets.cpu()

    plt.figure(figsize=(10, 4*num_images))
    for i in range(num_images):

        inp = inputs[i]
        tgt = targets[i]
        # Input image
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(inp.permute(1, 2, 0))  # (C,H,W) -> (H,W,C)
        plt.title('Input (Noisy)')
        plt.axis('off')

        # Output/target image
        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(tgt.permute(1, 2, 0))
        plt.title('Target (Clean)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_augumentation(ds, num_images = 1, n_image = 3):
    fig, axs = plt.subplots(num_images, 2, figsize=(8, num_images * 3))

    for i in range(num_images):
        noisy, clean = ds[n_image]
        noisy_np = noisy.permute(1, 2, 0).numpy()
        clean_np = clean.permute(1, 2, 0).numpy()

        axs[i, 0].imshow(clean_np)
        axs[i, 0].set_title(f"Clean Image {i + 1}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(noisy_np)
        axs[i, 1].set_title(f"Noisy Image {i + 1}")
        axs[i, 1].axis("off")

    fig.tight_layout()
    fig.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_parameters(n_stacks, in_channels = 3, C_ref = 64, Z_ref = 128, type='normal'):
    model = ChainedAutoencoder(n_stacks, in_channels, C_ref, Z_ref, type)
    print(f"Model with {n_stacks} stacks has {count_parameters(model):,} trainable parameters")

def train_model(model, loss_fcn, train_ds, val_ds, epochs):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Training on {device}")

    timex = time.time()
    history = {
        'loss': [],
        'val_loss': [],
        'time': [],
    }
    best_val_loss = float('inf')

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_ds, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for noisy, clean in loop:
            noisy_imgs, clean_imgs = noisy.to(device), clean.to(device)

            recon_imgs = model(noisy_imgs)

            loss = loss_fcn(recon_imgs, clean_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({'Train loss': train_loss / (loop.n + 1)})

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for noisy, clean in val_ds:
                noisy_imgs, clean_imgs = noisy.to(device), clean.to(device)
                recon_imgs = model(noisy_imgs)
                loss = loss_fcn(recon_imgs, clean_imgs)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_ds)
        avg_val_loss = val_loss / len(val_ds)
        time_epoch = time.time() - timex

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['time'].append(time_epoch)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f} - Time: {time_epoch:.2f}s")

    return model, history

def mse_loss(output, target):
    return F.mse_loss(output, target, reduction='mean')

def frequency_loss(predicted, target, p: float = 0.8) -> torch.Tensor:
    mse_spatial = F.mse_loss(predicted, target, reduction='mean')

    F_predicted = torch.fft.fft2(predicted, norm='ortho')
    F_target = torch.fft.fft2(target, norm='ortho')

    mag_predicted = torch.abs(F_predicted)
    mag_target = torch.abs(F_target)

    mse_freq = F.mse_loss(mag_predicted, mag_target, reduction='mean')

    return mse_spatial + p * mse_freq

def focal_frequency_loss(predicted, target, gamma: float = 1.5, reduction: str = 'mean') -> torch.Tensor:

    F_predicted = torch.fft.fft2(predicted, norm='ortho')
    F_target = torch.fft.fft2(target, norm='ortho')

    F_diff = F_predicted - F_target
    eps = 1e-8
    mag = torch.abs(F_diff) + eps
    weight = mag.pow(gamma)
    loss_map = weight * mag.pow(2)

    loss_per_sample = loss_map.view(loss_map.size(0), -1).sum(dim=1)

    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


class SmallAutoencoder(nn.Module):
    def __init__(self, image_channels, base_channels, latent_dim):
        super(SmallAutoencoder, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(),
        ])

        # latent space
        flat_dim = base_channels * 4 * 32 * 32

        self.fc_enc = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.decoders = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
        # Flatten and encode to latent space
        batch, c, h, w = x.shape
        x = x.view(batch, -1)
        z = self.fc_enc(x)
        # Decode from latent space
        x = self.fc_dec(z)
        x = x.view(batch, c, h, w)
        # Decoder path
        for decoder in self.decoders:
            x = decoder(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class UnetAutoencoder(nn.Module):
    def __init__(self, image_channels, base_channels, latent_dim):
        super().__init__()
        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(),
            ResidualBlock(base_channels),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(),
            ResidualBlock(base_channels * 2),
            nn.Dropout2d(0.2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(),
            ResidualBlock(base_channels * 4),
            nn.Dropout2d(0.2),
        )

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(base_channels*4*32*32, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, base_channels*4*32*32)

        # decoder
        self.dec3 = nn.Sequential(
            ResidualBlock(base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        self.dec1 = nn.Sequential(
            ResidualBlock(base_channels),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        z = self.fc_enc(self.flatten(e3))
        x_latent = self.fc_dec(z).view(e3.shape)

        d3 = self.dec3(e3 + x_latent)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)

        return d1






class ChainedAutoencoder(nn.Module):
    def __init__(self, num_aes, in_channels = 3, C_ref = 64, Z_ref = 128, type='normal'):
        super(ChainedAutoencoder, self).__init__()
        s = 1.0 / math.sqrt(num_aes)
        c = max(1, int(C_ref * s))
        z = max(1, int(Z_ref * s))
        if type == 'normal':
            self.aes = nn.ModuleList([
                SmallAutoencoder(in_channels, c, z) for _ in range(num_aes)
            ])
        elif type == 'unet':
            self.aes = nn.ModuleList([
                UnetAutoencoder(in_channels, c, z) for _ in range(num_aes)
            ])
        else:
            raise ValueError('type must be either normal or unet')

    def forward(self, x):
        out = x
        for aes in self.aes:
            out = aes(out)
        return out

class PariedImages(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.input_dir = os.path.join(root, 'output')
        self.output_dir = os.path.join(root, 'input')
        self.input_files = sorted(os.listdir(self.input_dir))
        self.output_files = sorted(os.listdir(self.output_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])

        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img

class PairedDSVariableEpoch(Dataset):
    def __init__(self, image_paths, transform=None):
        super().__init__()
        self.image_paths = glob.glob(image_paths + '/*.bmp')
        self.transform = transform
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)

        self.im_sz = self.settings['img_size']
        self.im_max_hz = self.settings['max_hz']
        self.im_gaussian_mu = self.settings['gaussian_mean']
        self.im_gaussian_std = self.settings['gaussian_std']
        self.im_holes_amount = self.settings['hole_amount']
        self.im_holes_radius = self.settings['hole_radius']
        self.im_resizing_policy = self.settings['resizing_policy']
        self.im_resizing_method = self.settings['resizing_method']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        output_img = ImageProcess(self.image_paths[idx])
        if output_img.original_image.shape != (*self.im_sz, 3):
            output_img.resize_image(self.im_sz, self.im_resizing_policy, self.im_resizing_method)

        max_hz = random.uniform(self.im_max_hz[0], self.im_max_hz[1])
        mu = random.uniform(self.im_gaussian_mu[0], self.im_gaussian_mu[1])
        sigma = random.uniform(self.im_gaussian_std[0], self.im_gaussian_std[1])
        hole_amount = random.randint(self.im_holes_amount[0], self.im_holes_amount[1])

        output_img.dist_blackholes(self.im_holes_radius[0], self.im_holes_radius[1], hole_amount)
        output_img.dist_lowpass(max_hz)
        output_img.dist_noise_gaussian(mu, sigma)

        out_img = cv2.cvtColor(output_img.original_image, cv2.COLOR_BGR2RGB)
        in_img = cv2.cvtColor(output_img.image, cv2.COLOR_BGR2RGB)


        out_img = Image.fromarray(out_img)
        in_img = Image.fromarray(in_img)

        if self.transform:
            in_img = self.transform(in_img)
            out_img = self.transform(out_img)

        return in_img, out_img

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = PairedDSVariableEpoch('db/dataset_preprocessed/train/input', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)

    inputs, outputs = next(iter(train_loader))
    show_pair(inputs, outputs, 3)
    show_augumentation(train_dataset, 3, 4)

    #for traing_parameter in training_parameters:
        #model = ChainedAutoencoder(*traing_parameter)
        #savename = '{}'
    #model = ChainedAutoencoder(1, 3, 32, 512)
    #model, history = train_model(model, mse_loss, train_loader, val_loader, 1)





