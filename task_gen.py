import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Define some parameters
batch_size = 100
num_epochs = 5  # Number of training epochs
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 5 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers

# Set up the dataset and dataloader
# transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(32),transforms.Normalize(0.5,0.5)])
# train_data = datasets.FashionMNIST(root = 'data', train = True, transform = transform, download = True)
# dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)



import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ClockDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        label_path = img_path.replace('.png', '.txt')

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            label = f.read().strip().split(':')
            hour, minute = int(label[0]), int(label[1])

        return image, torch.tensor([hour, minute])



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z of size B x nz x 1 x 1, we put this directly into a transposed convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: B x (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size: B x (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: B x (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
            # Size: B x 1 x 32 x 32
            nn.Tanh()
        )

    def forward(self, z, conditions):
        # 将条件信息与潜在向量连接
        conditions = conditions.unsqueeze(-1).unsqueeze(-1)
        conditions = conditions.expand(z.size(0), conditions.size(1), z.size(2), z.size(3))
        input = torch.cat((z, conditions), 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # state size. 1 x 32 x 32
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, conditions):
        # 将条件信息与图像数据连接
        conditions = conditions.unsqueeze(-1).unsqueeze(-1)
        conditions = conditions.expand(img.size(0), conditions.size(1), img.size(2), img.size(3))
        input = torch.cat((img, conditions), 1)
        return self.main(input)

def train():
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 创建数据集和数据加载器
    data_dir = 'datasets/sub_clock'
    dataset = ClockDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # for images, labels in dataloader:
    #     hour, minute = labels[:, 0], labels[:, 1]

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    print("Starting Training Loop...")
    # For each epoch
    # 训练循环
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader, 0):
            # 解包标签为小时和分钟
            hour, minute = labels[:, 0], labels[:, 1]
            # 将小时和分钟转换为独热编码
            # conditions = torch.cat((torch.eye(24)[hours], torch.eye(60)[minutes]), 1).to(device)
            condition = torch.zeros(1440)
            condition[hour * 60 + minute] = 1
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_images = images.to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_images,condition).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(z)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, condition.detach()).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


def perdict():
    # 随机生成 16 张指定时间的时钟图像
    times = np.random.randint(0, 24 * 60, size=4)
    z = torch.randn(16, nz, 1, 1, device=device)
    labels = torch.tensor([t for _ in range(4) for t in times]).to(device)
    with torch.no_grad():
        fake_images = netG(z, labels).cpu()
        fake_images = (fake_images + 1) / 2

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            axs[i, j].imshow(fake_images[idx].permute(1, 2, 0))
            axs[i, j].set_title(f"{times[i] // 60:02d}:{times[i] % 60:02d}")
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()




    # 在潜空间中对 4 个不同时间各生成 2 张随机时钟图像并插值
    times = np.random.randint(0, 24 * 60, size=4)
    z1 = torch.randn(4, nz, 1, 1, device=device)
    z2 = torch.randn(4, nz, 1, 1, device=device)
    labels = torch.tensor(times).to(device)

    with torch.no_grad():
        for t, label in zip(times, labels):
            interpolated = []
            for alpha in np.linspace(0, 1, 7):
                z = z1 + alpha * (z2 - z1)
                fake_img = netG(z, label.repeat(4, 1)).cpu()
                interpolated.append((fake_img + 1) / 2)

            interpolated = torch.cat(interpolated, dim=0)

            fig, axs = plt.subplots(1, 7, figsize=(14, 2))
            for j in range(7):
                axs[j].imshow(interpolated[j].permute(1, 2, 0))
                axs[j].axis('off')
            fig.suptitle(f"Interpolation at time {t // 60:02d}:{t % 60:02d}", size=16)
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator()
    netG = netG.to(device)

    netD = Discriminator()
    netD = netD.to(device)

    train()


