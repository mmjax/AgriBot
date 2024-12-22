import os
import zipfile

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from io import BytesIO


from predict import predict
from utils import generate_csv

# Инициализация FastAPI
app = FastAPI()

# Параметры
num_classes = 2  # Задайте количество классов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    prediction = predict(image)
    return JSONResponse(content={"prediction": prediction})


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
                results[filename] = prediction

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
