import math
import os
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.ao.quantization import quantize_dynamic
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from PIL import Image, ImageDraw, ImageFont


# dataloader
class ClockDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.images = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path)
        normalized_tensor = preprocess_image(image, augment=self.augment)
        label_path = img_path.replace('.png', '.txt')
        with open(label_path, 'r') as f:
            label = f.read().strip().split(':')
            hour, minute = float(label[0]), float(label[1])
            # Cycle Encoding
            encoded_hour = [math.sin(2 * math.pi * hour / 12), math.cos(2 * math.pi * hour / 12)]
            encoded_minute = [math.sin(2 * math.pi * minute / 60), math.cos(2 * math.pi * minute / 60)]

        return normalized_tensor, torch.tensor(encoded_hour + encoded_minute, dtype=torch.float32)


# channel attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


# spatial attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        att = 1 + self.channel_att(x) * x
        att = att + self.spatial_att(att)
        return att


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class CartoonClockCNN(nn.Module):
    def __init__(self):
        super(CartoonClockCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2)
        self.layer3 = self._make_layer(64, 128, 2)
        self.layer4 = self._make_layer(128, 256, 2)
        # self.layer5 = self._make_layer(256, 512, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.layer2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.layer3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.layer4(x)
        x = F.max_pool2d(x, 2, 2)
        # x = self.layer5(x)
        # x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def preprocess_image(image, augment=False, augment_prob=0.8):
    def add_noise(tensor):
        noise = torch.randn(tensor.size()) * 0.01
        tensor = tensor + noise
        return tensor

    transform_list = [transforms.Resize((64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    augment_transforms = [transforms.ColorJitter(),
                          transforms.RandomErasing(scale=(0.01, 0.01)),
                          transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.7, 1.2))]

    # If data enhancement is performed, some enhanced transformations are randomly selected and added to the transformation list
    if augment and random.random() < augment_prob:
        selected_augment_transforms = random.sample(augment_transforms, k=random.randint(1, len(augment_transforms)))
        transform_list.extend(selected_augment_transforms)
        # Adding Noise with Lambda Transformations
        transform_list.append(transforms.Lambda(add_noise))

    composed_transforms = transforms.Compose(transform_list)
    augmented_tensor = composed_transforms(image)

    return augmented_tensor


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Creating a Network Instance
    model = CartoonClockCNN().to(device)
    train_dataset = ClockDataset('/content/train')
    # train_dataset = ClockDataset('datasets/sub_clock')

    # Splite Dataset
    train_data, val_data = train_test_split(list(range(len(train_dataset))), test_size=0.2)
    train_subset = torch.utils.data.Subset(train_dataset, train_data)
    val_subset = torch.utils.data.Subset(train_dataset, val_data)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)

    # Loss function
    criterion = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

    num_epochs = 100
    train_losses = []
    all_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            all_losses.append(loss.item())
            running_loss += loss.item()

            if (i + 1) % 25 == 0:  # print every 25 batch
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Current Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        scheduler.step()
        if (epoch + 1) % 5 == 0:  # Validation loss is calculated every 5 epochs
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Avg Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Avg Loss: {train_loss:.4f}")

    print('Training finished')

    # Save the trained model
    model.eval()
    model_path = 'cartoon_clocks_cnn.pth'
    torch.save(model.state_dict(), model_path)
    print('Model Saved')

    # Plotting training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1, 5), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def predict(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CartoonClockCNN().to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(img_path)
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    output = output.cpu().detach().numpy()[0]

    # Decoding
    predicted_hour = int((math.atan2(output[1], output[0]) + 2 * math.pi) % (2 * math.pi) / (2 * math.pi) * 12)
    predicted_minute = int((math.atan2(output[3], output[2]) + 2 * math.pi) % (2 * math.pi) / (2 * math.pi) * 60)
    print(predicted_hour, predicted_minute)
    return predicted_hour, predicted_minute


if __name__ == '__main__':
    # train()

    model_path = 'cartoon_clocks_cnn.pth'
    img_path = "datasets/clocks_dataset/train/0003.png"
    # predict(img_path, model_path)


    # Starting Predict
    # model_path = '/content/cartoon_clocks_cnn.pth'
    # img_path = "/content/train/0964.png"
    predicted_hour, predicted_minute = predict(img_path, model_path)

    label_path = img_path.replace('.png', '.txt')
    with open(label_path, 'r') as f:
        label = f.read().strip().split(':')
        hour, minute = float(label[0]), float(label[1])
    image = Image.open(img_path)

    # Create an ImageDraw object to draw text on the image.
    draw = ImageDraw.Draw(image)

    # Defining fonts and font sizes
    # font = ImageFont.truetype('arial.ttf', size=30)
    font = ImageFont.load_default()

    # Plotting predicted values and true labels on images
    draw.text((10, 10), f"Predicted: {predicted_hour:02d}:{predicted_minute:02d}", font=font, fill=(255, 0, 0))
    draw.text((10, 40), f"Label: {int(hour):02d}:{int(minute):02d}", font=font, fill=(0, 255, 0))

    # Show image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
