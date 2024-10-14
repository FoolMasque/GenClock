import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, DDPMScheduler
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class CartoonClockDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.data_dir = data_dir
        self.images = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path)
        normalized_tensor = preprocess_image(image)
        label_path = img_path.replace('.png', '.txt')
        with open(label_path, 'r') as f:
            label = f.read().strip().split(':')
            hour, minute = int(label[0]), int(label[1])
            normalized_hour = hour / 11
            normalized_minute = minute / 59
        labels = torch.tensor([normalized_hour, normalized_minute])
        # print(normalized_tensor.shape) # torch.Size([3, 256, 256])

        normalized_label = (hour*60+minute)/779
        label_channel = torch.ones((1, normalized_tensor.shape[1], normalized_tensor.shape[2])) * normalized_label
        normalized_tensor = torch.cat((normalized_tensor, label_channel), dim=0)
        # print(normalized_tensor.shape)
        return normalized_tensor, labels


def preprocess_image(image):
    transform_list = [transforms.Resize((256, 256)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    composed_transforms = transforms.Compose(transform_list)
    augmented_tensor = composed_transforms(image)

    return augmented_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建 SummaryWriter 实例用于记录指标
    writer = SummaryWriter()

    # 定义训练循环

    train_subset = CartoonClockDataset('datasets/sub_clock')
    data_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    # model_path = '/content/drive/My Drive/datasets/stable-diffusion-v1-5/unet'
    model_path = 'E:/dev/datasets/stable-diffusion-v1-5/unet'
    unet = UNet2DConditionModel.from_pretrained(model_path).to(device)

    # 冻结前几层参数
    for name, param in unet.named_parameters():
        if "down_blocks" in name or "mid_block" in name:  # 冻结down_blocks和mid_block层的参数
            param.requires_grad = False

    # 检查冻结情况
    for name, param in unet.named_parameters():
        print(name, param.requires_grad)


    # scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    # 定义扩散调度器
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    num_epochs = 10
    lr = 1e-4
    optim = torch.optim.Adam(unet.parameters(), lr=lr)
    # 训练

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(data_loader):
            labels = labels.to(device)
            images = images.to(device)

            # 随机选择一个时间步
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
            # 添加噪声
            noise = torch.randn_like(images).to("cuda")
            print(images.shape)
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            print(noisy_images.shape)
            # 使用UNet模型进行预测
            model_output = unet(noisy_images, timesteps, labels).sample
            # 计算损失
            loss = nn.MSELoss(model_output, images)


            optim.zero_grad()
            loss.backward()
            optim.step()


            running_loss += loss.item()
            # 监控和记录训练指标
            if (i + 1) % 100 == 0:  # 每100个batch输出一次
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Current Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/train', avg_loss, epoch * len(data_loader) + i)
                running_loss = 0.0
    # 保存模型
    torch.save(unet.state_dict(), 'unet_weights.pth')
    # scheduler.save_pretrained('trained_model_scheduler')

def predict():
    # 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4")
    unet.load_state_dict(torch.load('unet_weights.pth'))
    model = unet.to("cuda")







    # 定义您要生成图像的条件(小时和分钟)
    example_hours = [0, 6, 12, 18]
    example_minutes = [0, 15, 30, 45]
    example_conditions = torch.tensor(
        [[hour / 23, minute / 59] for hour, minute in zip(example_hours, example_minutes)]).to(device)

    # 生成图像
    with torch.no_grad():
        samples = model.sample(condition=example_conditions)

    # 后处理和保存生成的图像
    for i, sample in enumerate(samples):
        sample = (sample / 2 + 0.5).clamp(0, 1)  # 调整到 [0, 1] 范围
        # sample = sample.permute(0, 2, 3, 1).squeeze()  # 调整通道维度
        sample = sample.permute(1, 2, 0).cpu().numpy()
        # save_image(sample, f'clock_{example_hours[i]}_{example_minutes[i]}.png')
        save_image(torch.tensor(sample).permute(2, 0, 1), f'clock_{example_hours[i]}_{example_minutes[i]}.png')


if __name__ == "__main__":
    train()
    # predict()