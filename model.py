import math
import time
from os import mkdir

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt


def check_and_make(modelpath):
    if not os.path.exists(modelpath):
        print('creating {} folder'.format(modelpath))
        mkdir(modelpath)
    else:
        print('{} folder detected, removing'.format(modelpath))
        os.rmdir(modelpath)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_parameters(n_stacks, in_channels = 3, C_ref = 64, Z_ref = 128):
    model = ChainedAutoencoder(n_stacks, in_channels, C_ref, Z_ref)
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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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


class ChainedAutoencoder(nn.Module):
    def __init__(self, num_aes, in_channels = 3, C_ref = 64, Z_ref = 128):
        super(ChainedAutoencoder, self).__init__()
        s = 1.0 / math.sqrt(num_aes)
        c = max(1, int(C_ref * s))
        z = max(1, int(Z_ref * s))

        self.aes = nn.ModuleList([
            SmallAutoencoder(in_channels, c, z) for _ in range(num_aes)
        ])

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



if __name__ == "__main__":


    training_parameters = [
        [1, 3, 32, 512],
        [1, 3, 64, 256],
        [1, 3, 128, 128],
        [1, 3, 256, 64],
        [2, 3, 32, 512],
        [2, 3, 64, 256],
        [2, 3, 128, 128],
        [2, 3, 256, 64],
        [3, 3, 32, 512],
        [3, 3, 64, 256],
        [3, 3, 128, 128],
        [3, 3, 256, 64],
        [4, 3, 32, 512],
        [4, 3, 64, 256],
        [4, 3, 128, 128],
        [4, 3, 256, 64],
    ]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = PariedImages('db/dataset_preprocessed/train', transform=transform)
    val_dataset = PariedImages('db/dataset_preprocessed/val', transform=transform)
    test_dataset = PariedImages('db/dataset_preprocessed/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    inputs, outputs = next(iter(train_loader))
    print(inputs.shape, outputs.shape)
    print(inputs[1][1].max())
    modelpath =  os.path.join(os.getcwd(), 'models')
    historypath = os.path.join(os.getcwd(), 'history')


    check_and_make(modelpath)
    check_and_make(historypath)

    show_pair(inputs, outputs, 3)
    i = 0
    for loss, loss_name in zip([mse_loss, frequency_loss], ['mse', 'frequency']):
        for training_parameter in training_parameters:
            i += 1
            savename = 'model_{}_{}_loss_{}stacks_{}colors_{}Csize_{}Zsise'.format(i, loss_name, *training_parameter)
            model = ChainedAutoencoder(*training_parameter)
            model, history = train_model(model, loss, train_loader, val_loader, 1)

            model_savename = os.path.join(modelpath, '{}.pth'.format(savename))
            history_savename = os.path.join(historypath, '{}.csv'.format(savename))

            torch.save(model.state_dict(), model_savename)
            df = pd.DataFrame.from_dict(history)
            df.insert(0, 'epoch', df.index + 1)
            df.to_csv(history_savename, index = False)

    #for traing_parameter in training_parameters:
        #model = ChainedAutoencoder(*traing_parameter)
        #savename = '{}'
    #model = ChainedAutoencoder(1, 3, 32, 512)
    #model, history = train_model(model, mse_loss, train_loader, val_loader, 1)





