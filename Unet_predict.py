import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import cv2


class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = UNetConvBlock(3, 64)  # 3 канала на входе для RGB изображений
        self.enc2 = UNetConvBlock(64, 128)
        self.enc3 = UNetConvBlock(128, 256)
        self.enc4 = UNetConvBlock(256, 512)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetConvBlock(512, 1024)

        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetConvBlock(1024, 512)
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetConvBlock(512, 256)
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetConvBlock(256, 128)
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetConvBlock(128, 64)

        self.conv_last = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

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

        return self.conv_last(dec1)

# Определите модель UNet и загрузите весы
model = UNet()
model.load_state_dict(torch.load('unet_model.pth',weights_only=True))  # Загрузка обученной модели
model.eval()  # Перевод модели в режим предсказания

# Определяем устройство (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Функция для загрузки изображения
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Добавляем размер батча (B, C, H, W)
    return image

image_number = input("Введите номер изображения (например, 15 для 'image (15).jpg'): ")

# Формируем путь к изображению на основе введенного номера
image_path = f'dataset/generated/image ({image_number}).jpg'  # Замените на путь к вашему изображению

# Преобразования (должны совпадать с теми, что использовались при обучении)
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Измените на тот же размер, что был на этапе обучения
    transforms.ToTensor()
])

# Загрузка и предсказание
image = load_image(image_path, transform)
image = image.to(device)

with torch.no_grad():
    output = model(image)
    prediction = torch.sigmoid(output)
    prediction = prediction.cpu().numpy()

# Постобработка предсказания (бинаризация маски)
threshold = 0.5
binary_mask = (prediction > threshold).astype(np.uint8)

# Загрузка исходного изображения для визуализации
original_image = Image.open(image_path).convert("RGB")
original_image_np = np.array(original_image)

# Поиск контуров на предсказанной маске
binary_mask_squeezed = binary_mask.squeeze()
contours, _ = cv2.findContours(binary_mask_squeezed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Наложение контуров на исходное изображение
image_with_contours = original_image_np.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)  # Зелёные контуры

output_image_path = 'Result/image.jpg'  # Укажите путь для сохранения
cv2.imwrite(output_image_path, cv2.cvtColor(image_with_contours, cv2.COLOR_RGB2BGR))

# Визуализация
plt.figure(figsize=(10, 5))
plt.imshow(image_with_contours)
plt.title("Предсказанная маска с контурами")
plt.axis('off')
plt.show()
