import re
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import time




def parse_address(df, from_col, neiro=0):
    # основная функция, которая создает нужные колонки и парсит в них адрес
    start_time = time.time()
    df = df.copy()
    df["Адрес"] = df[from_col].apply(cleen_string)
    df["Муниципалитет"] = df["Адрес"].apply(get_mo_re, neiro=neiro)
    print("Муниципалитет определен")
    # df["Населенный пункт"] = df["Адрес"].apply(get_nas_punkt)
    df["Населенный пункт"] = df.apply(lambda row: get_nas_punkt_new(row["Адрес"], row["Муниципалитет"]), axis=1)
    print("Населенный пункт определен")
    # df["Улица"] = df["Адрес"].apply(get_street)
    df["Улица"] = df.apply(lambda row: get_street_new(row["Адрес"], row["Муниципалитет"], row["Населенный пункт"]), axis=1)
    # df["Улица"] = df.apply(get_street_new, args=(2, 3), axis=1)
    print("Улица определена")
    df["Дом"] = df.apply(lambda row: get_home_new(row["Адрес"], row["Улица"]), axis=1)
    print("Дом определен")
    df["Квартира"] = df["Адрес"].apply(get_kvartira)
    print("Квартира определена \n")
    report = df[["Адрес", "Муниципалитет", "Населенный пункт", "Улица", "Дом", "Квартира"]].notna().sum() / df.shape[0] * 100

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Время выполнения: {execution_time} минут")

    print(report)
    df[["Адрес", "Муниципалитет", "Населенный пункт", "Улица", "Дом", "Квартира"]].notna().sum().plot(kind="barh")
    return df



def losk(string):
    # приводит к единому формату элементы адреса
    if string is None:
        return None
    else:
        string = str(string).strip()
        levels = [r"Г\.( *)+",
                  r"Г( )+",
                  r"ГОРОД ",
                  r"СЕЛО ",
                  r"С\.( *)+",
                  r"С( )+",
                  r"ПОСЕЛОК ",
                  r"П\.( *)+",
                  r"П( )+",
                  r"СТАНЦИЯ ",
                  r"СТ\.( *)+",
                  r"СТ( )+",
                  r"ХУТОР ",
                  r"Х\.( *)+",
                  r"Х( )+",
                  r"УЛИЦА ",
                  r"УЛ\.( *)+",
                  r"УЛ( )+",
                  r"ПЕРЕУЛОК ",
                  r"ПЕР\.( *)+",
                  r"ПЕР( )+",

                  ]
        replacement = ["Г. ", "Г. ", "Г. ",
                       "С. ", "С. ", "С. ",
                       "П. ", "П. ", "П. ",
                       "СТ. ", "СТ. ", "СТ. ",
                       "Х. ", "Х. ", "Х. ",
                       "УЛ. ", "УЛ. ", "УЛ. ",
                       "ПЕР. ", "ПЕР. ", "ПЕР. "]

        for pattern, repl in zip(levels, replacement):
            match = re.search(pattern, string, flags=re.IGNORECASE)
            if match:
                res = re.sub(pattern, repl, string, flags=re.IGNORECASE)
                return res
                break

        return string


# print(losk("С. ЖУРАВЛЕВКА"))
def cleen_string(string):
    # очищает строку от не нужных символов и переводит в верхний регистр
    if string is None:
        return None
    else:
        string = str(string).strip()
        pattern = r"[А-Яа-яA-Za-z0-9 _,.;:()«»\"'/-]+"
        match = re.findall(pattern, string)
        if match:
            res = " ".join(match)
            return res.upper()


def get_nas_punkt(string):
    # вытягивает из строки населенный пункт
    import re
    if string is None:
        return None
    else:
        string = str(string)
        if get_reverse_neiroset(string, model=model_reverse, tokenizer=tokenizer_reverse) == "вначале":
            levels = [r"Г\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от Г. до запятой
                      r"С\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от С. до запятой
                      r"СЕЛО [А-Яа-я -]+?(?=,|$)",  # ищет от СЕЛО до запятой
                      r"П\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от П. до запятой
                      r"\sПГТ\.[ ]{0,1}[А-Яа-я -]+?(?=\s|$|,)",
                      r"\sПГТ[ ]{0,1}[А-Яа-я -]+?(?=\s|$|,)",
                      r"Г\.[ ]{0,1}[А-Яа-я -]+?(?=\s)",
                      r"\sС\s[А-Яа-я -]+?(?=\s)",
                      r"\sСЕЛО\s[А-Яа-я -]+?(?=\s|$)",
                      r"\sПОСЕЛОК\s[А-Яа-я -]+(?=$|,)",
                      r"\Г\s[А-Яа-я -]+?(?=$|,)",
                      r"\ХУТОР\s[А-Яа-я -]+?(?=$|,)",
                      r"Х\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",
                      r"\СТАНЦИЯ\s[А-Яа-я -]+?(?=$|,)",
                      r"\sС\s[А-Яа-я -]+?(?=,|$)",
                      r"\sП\s[А-Яа-я -]+?(?=,|$)",
                      r"\sГ\s[А-Яа-я -]+?(?=,|$)",
                      ]
            for pattern in levels:
                match = re.findall(pattern, string, flags=re.IGNORECASE)
                if len(match) == 1:
                    res = losk(match[0])
                    return losk(res)

        else:
            levels = [r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ Г\.",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ Г\.",
                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ Г\s",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ Г\s",

                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ С\.",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ С\.",
                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ С\s",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ С\s",

                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ П\.",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ П\.",
                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ П\s",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ П\s",

                      r"\w+ ПГТ\.",
                      r"\w+ Г\.",
                      r"\w+ Г\.",
                      ]
            for pattern in levels:
                match = re.findall(pattern, string, flags=re.IGNORECASE)
                if len(match) == 1:
                    res = losk(match[0])
                    return losk(res)





# print(get_nas_punkt("БЕЛГОРОДСКАЯ ОБЛАСТЬ, Р-Н ВАЛУЙСКИЙ, С БИРЮЧ, УЛ ГАГАРИНА, Д 28"))



def get_street(string):
    # получает из строки улицу
    if string is None:
        return None
    else:
        string = str(string).strip()
        if get_reverse_neiroset(string, model=model_street_reverse, tokenizer=tokenizer_street_reverse) == "вначале":
            levels = [r"УЛ\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от УЛ. до запятой
                      r"УЛ\.[ ]{0,1}\d{0,2}[А-Яа-я -]+?(?=,|$)",  # ищет от УЛ. до запятой с цифрами в названии
                      r"\sУЛИЦА\s[А-Яа-я -]+?(?=,|$)",
                      r"\sУЛИЦА\s\d{0,2}[А-Яа-я -]+?(?=,|$)",
                      r"ПЕР\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от ПЕР. до запятой
                      r"ПЕР\.[ ]{0,1}\d{0,2}[А-Яа-я -]+?(?=,|$)",  # ищет от ПЕР. до запятой с цифрами в названии
                      ]
            for pattern in levels:
                match = re.findall(pattern, string, flags=re.IGNORECASE)
                if len(match) == 1:
                    return losk(match[0])

        else:
            levels = [r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ УЛ\.",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ УЛ\.",
                      r"(?<=\,)[ ]{0,1}[А-Яа-я -]+ УЛ\s",
                      r"(?<=\.)[ ]{0,1}[А-Яа-я -]+ УЛ\s",
            #
            #           r"УЛ\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от УЛ. до запятой
            #           r"УЛ\.[ ]{0,1}\d{0,2}[А-Яа-я -]+?(?=,|$)",  # ищет от УЛ. до запятой с цифрами в названии
            #           r"\sУЛИЦА\s[А-Яа-я -]+?(?=,|$)",
            #           r"\sУЛИЦА\s\d{0,2}[А-Яа-я -]+?(?=,|$)",
            #           r"ПЕР\.[ ]{0,1}[А-Яа-я -]+?(?=,|$)",  # ищет от ПЕР. до запятой
            #           r"ПЕР\.[ ]{0,1}\d{0,2}[А-Яа-я -]+?(?=,|$)",  # ищет от ПЕР. до запятой с цифрами в названии
                      ]
            for pattern in levels:
                match = re.findall(pattern, string, flags=re.IGNORECASE)
                if len(match) == 1:
                    return losk(match[0])

# =================================== парсинг НОВЫЙ ПОДХОД ===============================

def get_in_postgreSQL(name_table):
    import psycopg2
    from io import StringIO

    # для PostgreSQL
    user = 'data'
    port = 5432
    password = 'User123!@#'
    host = "10.100.33.6"
    database = 'sc_data'

    # Подключение к базе данных
    conn = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Создание курсора
    cur = conn.cursor()


    #Выполнение SQL-запроса
    cur.execute(f'SELECT * FROM {name_table}')

    #Получение результатов
    result = cur.fetchall()
    result = pd.DataFrame(result)
    # Закрытие курсора и соединения
    cur.close()
    conn.close()
    return result

spravochnik_street = get_in_postgreSQL("public.spravochnik_street")
spravochnik_street.columns = ['mo', 'nas_punkt', 'street', 'nas_punkt_suffix', 'street_suffix']
# spravochnik_street = spravochnik_street.apply(lambda x: x.astype(str).str.upper())
# spravochnik_street["nas_punkt_cleen"] = spravochnik_street["nas_punkt"].apply(lambda x: x.rsplit(maxsplit=1)[0])

# ============================================ НАСЕЛЕННЫЕ ПУНКТЫ =======================================
def get_nas_punkt_new(string, mo):
    '''
    Определяет населенный пункт с помощью справочника
    '''

    if (mo is None) | (string is None):
        return None
    else:
        mo = str(mo).upper()
        # print(mo)
        string = str(string).upper()
        if mo != "БЕЛГОРОД":
            end_index = string.find(mo) + len(mo)
            string = string[end_index :]
        # print(string)
        nas_punkt_list = []
        patterns = spravochnik_street
        nas_punkt = patterns.loc[(patterns["mo"] == mo)][["nas_punkt", "nas_punkt_suffix"]]
        nas_punkt = nas_punkt.drop_duplicates()
        # print(nas_punkt)
        for nas, suffix in zip(nas_punkt["nas_punkt"], nas_punkt["nas_punkt_suffix"]):
            # print(nas.rsplit(maxsplit=1)[0])
            match = re.search(nas, string, re.IGNORECASE)
            if match:
                nas_punkt_list.append([suffix, nas])

        # print(nas_punkt_list)
        if len(nas_punkt_list) == 1:
            res = nas_punkt_list[0]
            return ". ".join(res)
        elif len(nas_punkt_list) > 1:
            for item in nas_punkt_list:
                suffix = item[0]
                # print(suffix)
                match_suffix = re.search(fr'''(\s|,){suffix}(\s|\.)''', string, re.IGNORECASE)
                if match_suffix:
                    return ". ".join([suffix, item[1]])


# ===================================== УЛИЦЫ ====================================================
def get_street_new(string, mo, nas_punkt):
    '''
    Определяет улицу с помощью справочника
    '''

    if (mo is None) | (nas_punkt is None) | (string is None):
        return None
    else:
        mo = str(mo).upper()
        nas_punkt = str(nas_punkt).upper()
        nas_punkt = nas_punkt.split(maxsplit=1)[1]
        # print(nas_punkt)
        string = str(string).upper()

        end_index = string.find(nas_punkt) + len(nas_punkt)
        string = string[end_index :]

        street_list = []
        patterns = spravochnik_street
        streets = patterns.loc[(patterns["mo"] == mo) & (patterns["nas_punkt"] == nas_punkt)][["street", "street_suffix"]]
        streets = streets.drop_duplicates()
        for street, suffix in zip(streets["street"], streets["street_suffix"]):
            # print(pat)
            match = re.search(street, string, re.IGNORECASE)
            match_suffix = re.search(fr'''(\s|,){suffix}(\s|\.)''', string, re.IGNORECASE)
            if match:
                if match_suffix:
                    street_list.append([suffix, street])

        # print(street_list)
        if len(street_list) == 1:
            res = street_list[0]
            return ". ".join(res)
        elif len(street_list) > 1:
            for item in street_list:
                suffix = item[0]
                # print(suffix)
                match_suffix = re.search(fr'''(\s|,){suffix}(\s|\.)''', string, re.IGNORECASE)
                if match_suffix:
                    return ". ".join([suffix, item[1]])


# =================================================== ДОМА ===================================

def get_home_new(string, street):
    '''
        Определяет дом с помощью регулярного выражения
        '''

    if (string is None) | (street is None):
        return None
    else:
        string = str(string).upper()
        street = street.split(maxsplit=1)[1]
        end_index = string.find(street) + len(street)
        string = string[end_index:]
        # print(string)

        levels = [
        # r"(?<=Д\.|Д\s)[ ]{0,1}\d{1,3}[А-я]{0,1}[/-]{0,1}[А-я]{0,1}\d{0,3}(?=,|$)",
        #           # ищет от Д. до запятой либо пробела либо конца строки
        #           r"(?<=\,)[ ]{0,1}\d{1,3}[А-я]{0,1}[/-]{0,1}[А-я]{0,1}\d{0,3}(?=,|$)"
                  # то же самое, только со слешем между цифрами
                    r"\d+[А-я0-9/-]{0,5}(?=,|\s|$)"
                    # r"(?<=Д )\d{1,3}[А-я0-9/-]{0,5}(?=,|$)",
                    # r"(?<=Д.)[\s]{0,2}\d{1,3}[А-я0-9/-]{0,5}(?=,|$)",
                    # r"(?<= )\d{1,3}[А-я0-9/-]{0,5}(?= КВ. | К. |$)"
                  ]
        for pattern in levels:
            match = re.search(pattern, string, flags=re.IGNORECASE)
            # print(match)
            if match:
                # print(match[0])
                return match.group().strip()


# =============================================================================================


def get_home(string):
    # получает из строки номер дома
    if string is None:
        return None
    else:
        string = str(string).strip()
        levels = [r"(?<=Д\.|Д\s)[ ]{0,1}\d{1,3}[А-Яа-я/]{0,1}\d{0,3}?(?=,|\s|$)",
                  # ищет от Д. до запятой либо пробела либо конца строки
                  r"(?<=ДОМ\s)[ ]{0,1}\d{1,3}[А-Яа-я/]{0,1}\d{0,3}?(?=,|\s|$)",
                  # то же самое, только со слешем между цифрами
                  ]
        for pattern in levels:
            match = re.findall(pattern, string, flags=re.IGNORECASE)
            if len(match) == 1:
                return match[0].strip()
                break


# print(get_home("БЕЛГОРОДСКАЯ ОБЛАСТЬ, Р-Н БЕЛГОРОДСКИЙ, СТАНЦИЯ СТАРАЯ НАУМОВКА, Д. 1"))
def get_kvartira(string):
    # получает из строки номер квартиры
    if string is None:
        return None
    else:
        string = str(string).strip()
        levels = [r"(?<=КВ\.)[ ]{0,1}\d{1,3}[А-Яа-я]{0,1}?(?=[,. ])",  # ищет от КВ. до запятой, точки или пробела
                  r"(?<=КВ\.)[ ]{0,1}\d{1,3}[А-Яа-я]{0,1}?$",  # ищет от КВ. до конца строки
                  ]
        for pattern in levels:
            match = re.findall(pattern, string, flags=re.IGNORECASE)
            if len(match) == 1:
                return match[0].strip()
                break


def get_comment(string):
    # получает из строки комментарий вконце
    if string is None:
        return None
    else:
        string = str(string).strip()
        levels = [r"\(.+\)$",  # ищет круглые скобки вконце и их содержимое
                  r"[A-Z].+$",  # ищет от английский букв до конца строки
                  ]
        for pattern in levels:
            match = re.findall(pattern, string, flags=re.IGNORECASE)
            if len(match) == 1:
                return match[0].strip(r"[ ()]")
                break


# ========================================= МУНИЦИПАЛИТЕТЫ ======================================
def get_mo_re(string, neiro=0):
    '''
    Определяет муниципалитет с помощью регулярных выражений
    если neiro = 0, нейросеть не применяется
    если neiro = 1, нейросеть применяется только если нашло несколько мо в строке
    если neiro = 2, нейросеть применяется если совпадений не нашло вообще
    '''
    import re
    if string is None:
        return None
    else:
        string = str(string)
        mo = []
        patterns = {1: ['Алексеевский', r'(г. |г.|г )Алексеевка'],
                    2: ['город Белгород', r'(\s|^|\.)Белгород(?:,|\s|$)'],
                    3: [r'Белгородский[ ,]', 'Дубовое', ' Майский', "Северный",
                        "Таврово", "Новосадовый", "Политотдельский"],
                    4: ['Борисовский', 'Борисовка'],
                    5: ['Валуйский', 'Валуйки'],
                    6: ['Вейделевский', "Вейделевка"],
                    7: ["Волоконовский", "Волоконовка"],
                    8: [r'Грайворон(?:,|\s|$)', "Грайворонский"],
                    9: [r'(?<!улица)(?<!ул\.)(?<!ул)\s+Губкин', r"^Губкин", r"Губкин(?:,|\s|$|\sг)"],
                    10: ['Ивня'],
                    11: ['Корочанский', r"Короча(?:,|\s|$)"],
                    12: ['Красненский'],
                    13: ['Краснояружский', "Красная Яруга"],
                    14: ['Красногвардейский', "Красная Гвардия", r"Бирюч(?=\s|$|,)"],
                    15: ['Новооскольский', 'Новый Оскол'],
                    16: ['Прохоровский', 'Прохоровка'],
                    17: ['Ракитянский', 'Ракитное'],
                    18: ['Ровеньский', "Ровеньки"],
                    19: ['Старооскольский', 'Старый Оскол'],
                    20: ['Чернянский', 'Чернянка'],
                    21: ['Шебекинский', "Шебекино", "Маслова Пристань", "Муром", "Большетроицкое",
                         "Боровское", "Нижнее Березово", "Маломихайловка", "Белянка", "Червона Дибровка"
                                                                                      "Новая Таволжанка", "Максимовка"],
                    22: ['Яковлевский', r'(г. |г.|г |\s)Строитель(?:,|\s|$)', "Луханино", "Ольховка",
                         "Крапивенские дворы"]}

        mo_list = [None,
                   'Алексеевский',
                   'Белгород',
                   'Белгородский район',
                   'Борисовский',
                   'Валуйский',
                   'Вейделевский',
                   'Волоконовский',
                   'Грайворонский',
                   'Губкинский',
                   'Ивнянский',
                   'Корочанский',
                   'Красненский',
                   'Краснояружский',
                   'Красногвардейский',
                   'Новооскольский',
                   'Прохоровский',
                   'Ракитянский',
                   'Ровеньский',
                   'Старооскольский',
                   'Чернянский',
                   'Шебекинский',
                   'Яковлевский']

        for id, pattern in patterns.items():
            for pat in pattern:
                # print(pat)
                match = re.search(pat, string, re.IGNORECASE)
                if match:
                    mo.append(id)
                    break
        # print(mo)
        if len(mo) == 1:
            mo_id = mo[0]
            return mo_list[mo_id]
        elif (len(mo) > 1) and (neiro == 1):  # если нашло больше одного, применяем нейросеть
            string = str(string.lower())
            res = get_mo_neiroset(string)
            return res
        elif neiro == 2:  # если вообще не нашло, применяем нейросеть
            string = str(string.lower())
            res = get_mo_neiroset(string)
            return res


# ================================= нейросеть =============================================

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn

# загружаем модель
model_path = r"C:\Users\svetlichnyy_av\PycharmProjects\MO_нейросеть\обученная модель\rubert-tiny-sentiment-balanced"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def get_mo_neiroset(string):
    '''
      принимает текст с адресом,
      возвращает название муниципалитета
      '''

    mo = [np.nan,
          'Алексеевский',
          'Белгород',
          'Белгородский район',
          'Борисовский',
          'Валуйский',
          'Вейделевский',
          'Волоконовский',
          'Грайворонский',
          'Губкинский',
          'Ивнянский',
          'Корочанский',
          'Красненский',
          'Краснояружский',
          'Красногвардейский',
          'Новооскольский',
          'Прохоровский',
          'Ракитянский',
          'Ровеньский',
          'Старооскольский',
          'Чернянский',
          'Шебекинский',
          'Яковлевский'
          ]

    text = string.lower().strip()

    # токенизация текста
    tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=30)

    # Применение модели
    outputs = model(**tokenized_text)

    # Обработка результатов
    predictions = outputs.logits.argmax(dim=1)
    name_label = mo[predictions]

    return name_label


# ====================================================================================================


# загружаем модель которая определяет с какой стороны стоит тип населенного пункта
model_reverse_path = r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address\revers_neiroset\обученная модель\rubert-tiny-sentiment-balanced"
model_reverse = AutoModelForSequenceClassification.from_pretrained(model_reverse_path)
tokenizer_reverse = AutoTokenizer.from_pretrained(model_reverse_path)


# загружаем модель которая определяет с какой стороны стоит тип улицы
model_street_reverse_path = r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address\revers_neiroset\обученная модель\street"
model_street_reverse = AutoModelForSequenceClassification.from_pretrained(model_reverse_path)
tokenizer_street_reverse = AutoTokenizer.from_pretrained(model_reverse_path)


def get_reverse_neiroset(text, model, tokenizer):
    '''
      принимает текст с адресом,
      возвращает с какой стороны стоит тип населенного пункта
      '''

    cl = [
          'вначале',
          'вконце'
          ]

    text = text.lower().strip()

    # токенизация текста
    tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=14)

    # Применение модели
    outputs = model(**tokenized_text)

    # Обработка результатов
    predictions = outputs.logits.argmax(dim=1)
    name_label = cl[predictions]

    return name_label

# ===================================================================================================




