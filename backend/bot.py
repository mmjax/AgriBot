import os
import zipfile
from io import BytesIO
from tempfile import TemporaryDirectory

from PIL import Image
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import FSInputFile
from aiogram.fsm.storage.memory import MemoryStorage

from backend.predict import predict
from backend.utils import generate_csv
from dotenv import load_dotenv
load_dotenv()

# Настройки бота
API_TOKEN = os.getenv("API_TOKEN")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


def process_archive(archive_path: str, save_dir: str):
    """Обработка архива с изображениями"""
    os.makedirs(save_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(save_dir)

    results = {}
    for root, _, files in os.walk(save_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, filename)
                image = Image.open(file_path)
                prediction = predict(image)
                results[filename] = prediction

    return results


@dp.message(F.text.in_({"start", "help"}))
async def send_welcome(message: types.Message):
    await message.answer("Привет! Отправь мне изображение для классификации или архив с изображениями для обработки.")


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    photo_bytes = BytesIO()
    await bot.download(photo, destination=photo_bytes)
    photo_bytes.seek(0)

    image = Image.open(photo_bytes)
    prediction = predict(image)
    await message.answer(f"Предсказание: {prediction}")


@dp.message(F.document)
async def handle_document(message: types.Message):
    document = message.document

    if document.file_name.lower().endswith(".zip"):
        with TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, document.file_name)
            await bot.download(document, destination=archive_path)

            results = process_archive(archive_path, temp_dir)
            csv_path = os.path.join(temp_dir, "results.csv")
            generate_csv(results, csv_path)

            await message.answer("Обработка завершена. Результаты в файле:")
            await message.answer_document(FSInputFile(csv_path))
    elif document.file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_bytes = BytesIO()
        await bot.download(document, destination=image_bytes)
        image_bytes.seek(0)
        image = Image.open(image_bytes)

        prediction = predict(image)
        await message.answer(f"Предсказание: {prediction}")
    else:
        await message.answer("Этот формат файла не поддерживается.")


async def main():
    # Запуск бота
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
