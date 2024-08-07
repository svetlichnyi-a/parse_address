from parse_address import *
import pandas as pd

print("start")

PATH = r"files_for_parsing/TAP_Diagn_week20231221.csv"  # путь к файлу
COL = "Адрес_пациента_неформализованный"  # колонка которую нужно распарсить

df = pd.read_csv(PATH, encoding="utf-8", sep=",")  # загружаем таблицу.

parsed = parse_address(df, COL, neiro=1, add_mo_norm=1)  # парсим и сохраняем результат в переменную
parsed.to_excel("PARSED/TAP20231201_or.xlsx", index=False)  # сохраняем результат в эксель

print("end")