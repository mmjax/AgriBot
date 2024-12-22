def generate_csv(results: dict, csv_path: str):
    """Генерация CSV-файла с результатами"""
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_file.write("Filename,Prediction\n")
        for filename, prediction in results.items():
            csv_file.write(f"{filename},{prediction}\n")
