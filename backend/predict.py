import torch
from torch import nn

from torchvision import transforms
from torchvision import models
from PIL import Image


num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


# Модель ResNet
class ResNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)


# Загрузка модели
model = ResNetClassifier(num_classes).to(device)
model.load_state_dict(torch.load("resnet_model.ckpt", map_location=device))
model.eval()

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    """Прогноз для одиночного изображения"""
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return "weed" if predicted.item() == 1 else "non weed"
