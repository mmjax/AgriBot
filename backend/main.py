import csv
import os
import zipfile
from tempfile import TemporaryDirectory

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision import models
import torch.nn as nn

# Инициализация FastAPI
app = FastAPI()

# Параметры
num_classes = 2  # Задайте количество классов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


# Загрузка обученной модели
class ResNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)


# Создаем модель и загружаем сохраненные веса
model = ResNetClassifier(num_classes).to(device)
model.load_state_dict(torch.load("resnet_model.ckpt"))

model.eval()

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    prediction = predict(image)
    result = "weed" if prediction == 1 else "non weed"
    return JSONResponse(content={"prediction": result})


@app.post("/predict-archive/")
async def predict_archive(file: UploadFile = File(...)):
    # Папка для сохранения файлов
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем загруженный архив
    archive_path = os.path.join(save_dir, file.filename)
    with open(archive_path, "wb") as f:
        f.write(await file.read())

    # Распаковка архива
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(save_dir)

    # Прогнозируем для каждого изображения
    results = {}
    for root, _, files in os.walk(save_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, filename)
                image = Image.open(file_path)
                prediction = predict(image)
                result = "weed" if prediction == 1 else "non weed"
                results[filename] = result

    # Генерация CSV
    csv_path = os.path.join(save_dir, "results.csv")
    generate_csv(results, csv_path)

    # Возвращаем результаты и ссылки на файлы
    return JSONResponse(content={
        "results": results,
        "csv_url": f"/download-csv/?path={csv_path}"
    })


@app.get("/download-csv/")
async def download_csv(path: str):
    return FileResponse(path, media_type="text/csv", filename="results.csv")


def generate_csv(results: dict, csv_path: str):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Prediction"])
        for filename, prediction in results.items():
            writer.writerow([filename, prediction])
