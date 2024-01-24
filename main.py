from parse_address import *
import pandas as pd

print("start")

PATH = "на парсинг.xlsx"  # путь к файлу
COL = "Адрес_пациента_формализованный"  # колонка которую нужно распарсить

df = pd.read_excel(PATH, skiprows=0)  # загружаем таблицу.

parsed = parse_address(df, COL, neiro=1, add_mo_norm=1)  # парсим и сохраняем результат в переменную
parsed.to_excel("parsed.xlsx", index=False)  # сохраняем результат в эксель

print("end")