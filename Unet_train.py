import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Определение датасета
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Загрузка изображения и маски
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # Маска загружается как цветное изображение

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Извлекаем зеленый канал из маски
        mask = mask[1, :, :]  # Зеленый канал, который будет бинарной маской (0 или 1)

        # Преобразуем значения маски в 0 и 1
        mask = (mask > 0.5).float()  # Превращаем в бинарную маску (значения 0 и 1)

        # Добавляем размер канала для маски (1, H, W)
        mask = mask.unsqueeze(0)

        return image, mask

# Трансформации для изображений и масок
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Путь к данным
image_dir = "dataset_not_marked"
mask_dir = "dataset_marked"

# Создание датасета и загрузчика данных
train_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Определение блока сверток
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Определение архитектуры UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = UNetConvBlock(3, 64)  # 3 канала на входе для RGB изображений
        self.enc2 = UNetConvBlock(64, 128)
        self.enc3 = UNetConvBlock(128, 256)
        self.enc4 = UNetConvBlock(256, 512)

        # Maxpool для downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNetConvBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetConvBlock(128, 64)

        # Выходной слой
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.conv_last(dec1)

# Инициализация модели и использование GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Настройка функции потерь и оптимизатора
criterion = nn.BCEWithLogitsLoss()  # Для бинарной сегментации (целевые значения 0 или 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Очистка градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)

        # Вычисление потерь
        loss = criterion(outputs, masks)

        # Обратное распространение
        loss.backward()

        # Обновление параметров
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

# Сохранение обученной модели
torch.save(model.state_dict(), 'unet_model.pth')
print("Обучение завершено.")
