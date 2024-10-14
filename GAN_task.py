import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# Define some parameters
batch_size = 100
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 50  # Number of training epochs
lrG = 1e-4
lrD = 4e-4
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
nc = 3  # 输出图像通道数




class ClockDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.images = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        composed_transforms = transforms.Compose([transforms.Resize((64, 64)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
                                                  ])
        normalized_tensor = composed_transforms(image)

        label_path = img_path.replace('.png', '.txt')
        with open(label_path, 'r') as f:
            label = f.read().strip().split(':')
            hour, minute = label[0], label[1]
            normalized_hour = float(int(hour) / 11 * 2 - 1)
            normalized_minute = float(int(minute) / 59 * 2 - 1)
            normalized_label = torch.tensor([normalized_hour, normalized_minute], dtype=torch.float32)
        return normalized_tensor, normalized_label


# Set up the dataset and dataloader
dataset = ClockDataset('datasets/clocks_dataset/train')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + 2, ngf * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, input, labels):
        # Expand labels to [batch_size, 2, 1, 1]
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.expand(-1, -1, 1, 1)
        # Combine noise vector and labels
        combined_input = torch.cat([input, labels], dim=1)
        return self.main(combined_input)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # state size. 1 x 32 x 32
            nn.Conv2d(nc + 2, ndf, 4, 2, 1, bias=False),
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

    def forward(self, input, labels):
        # Expand labels to [batch_size, 2, height, width]
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.expand(-1, -1, input.shape[2], input.shape[3])
        # Combine images and labels
        combined_input = torch.cat([input, labels], 1)
        return self.main(combined_input)


def train():
    netG = Generator(nz, ngf, nc)
    netG = netG.to(device)

    netD = Discriminator(ndf, nc)
    netD = netD.to(device)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    # Learning rate schedulers
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (images, labels) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_images = images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            # 为了防止判别器过于强大，对输入数据添加少量噪声
            real_images += 0.01 * torch.randn_like(real_images)
            # Forward pass real batch through D
            # 获取判别器输出并调整标签形状
            output = netD(real_images, labels).view(-1)

            real_label_tensor = torch.full_like(output, real_label, dtype=torch.float, device=device)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_label_tensor)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            # 将噪声向量限制在[-1, 1]的范围内
            z = torch.clamp(z, -1, 1)
            # Generate fake image batch with G
            fake_images = netG(z, labels)
            # Classify all fake batch with D
            output = netD(fake_images.detach(), labels).view(-1)
            fake_label_tensor = torch.full_like(output, fake_label, dtype=torch.float, device=device)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_label_tensor)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # for p in netD.parameters():
            #     p.data.clamp_(-opt.clip_value, opt.clip_value)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # for _ in range(2):  # Update G more frequently
            if i % 5 == 0:
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake_images, labels).view(-1)
                # Calculate G's loss based on this output
                # fake labels are real for generator cost
                # real_label_tensor = torch.full((output.size(0),), 1, dtype=torch.float, device=device)  # 生成器目标是让判别器认为生成的图像是真实的
                real_label_tensor = torch.empty((output.size(0),), dtype=torch.float, device=device).uniform_(0.9, 1.0)
                errG = criterion(output, real_label_tensor)
                # Calculate gradients for G
                errG.backward(retain_graph=True)
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Update learning rates
        schedulerD.step()
        schedulerG.step()
    print("Train Finished.")
    print("Strat Save Models...")
    # 保存模型
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
    print("Models Saved")

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def generate_images_for_times(netG, times, num_samples=4):
    images = []
    for time in times:
        hour, minute = time
        normalized_hour = float(int(hour) / 11 * 2 - 1)
        normalized_minute = float(int(minute) / 59 * 2 - 1)
        labels = torch.tensor([normalized_hour, normalized_minute], dtype=torch.float32, device=device).view(1, 2)

    for _ in range(num_samples):
        noise = torch.randn(1, nz, 1, 1, device=device)
        # 将噪声向量限制在[-1, 1]的范围内
        noise = torch.clamp(noise, -1, 1)
        with torch.no_grad():
            fake_image = netG(noise, labels).cpu()
        images.append(fake_image.squeeze())

    grid = make_grid(images, nrow=num_samples, normalize=True)
    return grid


def generate_random_samples(netG, num_samples=8):
    images = []
    for _ in range(num_samples):
        noise = torch.randn(1, nz, 1, 1, device=device)
        # 将噪声向量限制在[-1, 1]的范围内
        noise = torch.clamp(noise, -1, 1)
        # 随机生成时间标签
        random_hour = torch.randint(0, 12, (1,)).item() / 11.0 * 2 - 1  # 生成0到11之间的整数并归一化
        random_minute = torch.randint(0, 60, (1,)).item() / 59.0 * 2 - 1  # 生成0到59之间的整数并归一化
        labels = torch.tensor([random_hour, random_minute], dtype=torch.float32, device=device).view(1, 2)  # 随机时间标签
        with torch.no_grad():
            fake_image = netG(noise, labels).cpu()
        images.append(fake_image.squeeze())

    grid = make_grid(images, nrow=4, normalize=True)
    return grid


def interpolate_images(netG, start_latent, end_latent, num_steps, label):
    interpolated_images = []
    for alpha in torch.linspace(0, 1, num_steps):
        interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
        with torch.no_grad():
            fake_image = netG(interpolated_latent, label).cpu()
        interpolated_images.append(fake_image.squeeze())
    return interpolated_images


def generate_interpolated_images_for_time(netG, time, num_interpolations=5):
    hour, minute = time
    normalized_hour = float(int(hour) / 11 * 2 - 1)
    normalized_minute = float(int(minute) / 59 * 2 - 1)
    labels = torch.tensor([normalized_hour, normalized_minute], dtype=torch.float32, device=device).view(1, 2)

    # 随机生成两个噪声向量
    start_latent = torch.randn(1, nz, 1, 1, device=device)
    # 将噪声向量限制在[-1, 1]的范围内
    start_latent = torch.clamp(start_latent, -1, 1)
    end_latent = torch.randn(1, nz, 1, 1, device=device)
    end_latent = torch.clamp(end_latent, -1, 1)

    # 进行插值并生成图像
    images = interpolate_images(netG, start_latent, end_latent, num_interpolations + 2, labels)
    grid = make_grid(images, nrow=num_interpolations + 2, normalize=True)
    return grid


def predict():
    # Define some parameters

    # train()
    netG = Generator(nz, ngf, nc)
    netG = netG.to(device)
    netG.load_state_dict(torch.load('generator.pth', map_location=torch.device(device)))
    netG.eval()

    # 生成4×4网格图像
    times = [(10, 30), (14, 45), (18, 15), (22, 0)]  # 随机选择的四个时间
    grid = generate_images_for_times(netG, times)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()

    # 生成8个随机样本的图像
    random_grid = generate_random_samples(netG)

    plt.figure(figsize=(8, 8))
    plt.imshow(random_grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()

    # 插值
    times = [(10, 30), (11, 45), (6, 15), (3, 0)]  # 随机选择的四个时间
    fig, axes = plt.subplots(len(times), 1, figsize=(15, 15))

    for i, time in enumerate(times):
        grid = generate_interpolated_images_for_time(netG, time)
        axes[i].imshow(grid.permute(1, 2, 0).numpy())
        axes[i].axis('off')
        axes[i].set_title(f"Interpolation for time {time[0]:02d}:{time[1]:02d}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train()
    # predict()
