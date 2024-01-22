import re
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import time

mo_norm = pd.read_excel(r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address\муниципалитеты.xlsx")


# ================================= ОСНОВНАЯ ФУНКЦИЯ ======================================
def parse_address(df, from_col, neiro=0, add_mo_norm=True):
    """
    :param df: датафрейм, в котором находится колонка с адресами
    :param from_col: имя колонки с адресами, которую нужно распарсить
    :param neiro: использовать ли нейросеть для определения мо, если обычный метод не справляется
    :param add_mo_norm: добавлять колонку с полным описанием мо
    :return: датафрейм с новыми колонками, в которых записан результат парсинга адресов
    """
    # основная функция, которая создает нужные колонки и парсит в них адрес
    start_time = time.time()
    df = df.copy()
    df["Адрес"] = df[from_col].apply(cleen_string)
    df["Муниципалитет"] = df["Адрес"].apply(get_mo_re, neiro=neiro)
    df[["Населенный пункт", "Улица", "Дом", "Квартира"]] = df.apply(main_func, axis=1, result_type='expand')

    # =================================== отчет ===============================================

    report_proc = round(
        df[["Адрес", "Муниципалитет", "Населенный пункт", "Улица", "Дом", "Квартира"]].isna().sum() / df.shape[
            0] * 100, 2)
    report_abs = df[["Адрес", "Муниципалитет", "Населенный пункт", "Улица", "Дом", "Квартира"]].isna().sum()
    report = pd.concat([report_abs, report_proc.to_frame()], axis=1)
    report.columns = ["количество", "процент"]
    print(report)

    # ==================================== график =============================================

    df[["Адрес", "Муниципалитет", "Населенный пункт", "Улица", "Дом", "Квартира"]] \
        .notna().sum().plot(kind="barh", figsize=(3, 2))

    # ================== добавление нормализованного муниципалитета ==========================

    if add_mo_norm:
        df = df.merge(mo_norm, how="left", on="Муниципалитет")

    # ============================ время выполнения ==========================================

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"\nВремя выполнения: {execution_time} мин.")
    return df


# ====================================================================================

def main_func(row):
    nas_punkt = get_nas_punkt_new(row["Адрес"], row["Муниципалитет"])
    street = get_street_new_opt(row["Адрес"], row["Муниципалитет"], nas_punkt)
    home = get_home_new(row["Адрес"], street)
    kvartira = get_kvartira_new(row["Адрес"], street)
    return (nas_punkt, street, home, kvartira)


# =====================================================================================

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


# ========================================= МУНИЦИПАЛИТЕТ ======================================
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
        patterns = {1: ['Алексеевский', 'Алексеевского', r'(г. |г.|г |ГОРОД |ГОР )Алексеевка'],
                    2: ['город Белгород', r'(\s|^|\.|,)Белгород(?:,|\s|$)'],
                    3: [r'Белгородский[ ,]', 'Дубовое', ' Майский', "Северный",
                        "Таврово", "Новосадовый", "Политотдельский"],
                    4: ['Борисовский', 'Борисовка', 'Борисовского'],
                    5: ['Валуйский', 'Валуйки', 'Валуйского'],
                    6: ['Вейделевский', "Вейделевка", "Вейделевского"],
                    7: ["Волоконовский", "Волоконовка", "Волоконовского"],
                    8: [r'Грайворон(?:,|\s|$)', "Грайворонский", "Грайворонского", "Козинка"],
                    9: [r'(?<!улица)(?<!ул\.)(?<!ул)\s+Губкин', r"^Губкин", r"Губкин(?:,|\s|$|\sг)", "Губкинского"],
                    10: ['Ивня', "Ивнянского"],
                    11: ['Корочанский', r"Короча(?:,|\s|$)", "Корочанского"],
                    12: ['Красненский', "Красненского"],
                    13: ['Краснояружский', "Красная Яруга", "Краснояружского"],
                    14: ['Красногвардейский', "Красногвардейского", "Красная Гвардия", r"Бирюч(?=\s|$|,)"],
                    15: ['Новооскольский', 'Новый Оскол', "Новооскольского"],
                    16: ['Прохоровский', 'Прохоровка', "Прохоровского"],
                    17: ['Ракитянский', 'Ракитное', "Ракитянского"],
                    18: ['Ровеньский', "Ровеньки", "Ровеньского"],
                    19: ['Старооскольский', 'Старый Оскол', "Старооскольского"],
                    20: ['Чернянский', 'Чернянка', "Чернянского"],
                    21: ['Шебекинский', "Шебекино", "Шебекинского", "Маслова Пристань", "Муром", "Большетроицкое",
                         "Боровское", "Нижнее Березово", "Маломихайловка", "Белянка", "Червона Дибровка",
                         "Новая Таволжанка", "Максимовка"],
                    22: ['Яковлевский', "Яковлевского", r'(г. |г.|г |\s)Строитель(?:,|\s|$)', "Луханино", "Ольховка",
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
            res = mo_list[mo_id]
            # print("1")
            return res
        elif (len(mo) > 1) and (neiro == 1):  # если нашло больше одного, применяем нейросеть
            string = str(string.lower())
            res = get_mo_neiroset(string)
            # print("2")
            return res
        elif neiro == 2:  # если вообще не нашло, применяем нейросеть
            string = str(string.lower())
            res = get_mo_neiroset(string)
            # print("3")
            return res
        else:
            match = re.search("Белгородская", string, re.IGNORECASE)
            if match:
                return get_mo_spr(string)  # попытка определить муниципалитет по справочнику

# ============================================ МУНИЦИПАЛИТЕТ ПО СПРАВОЧНИКУ ==========================================

def get_mo_spr(string):
    '''
    Определяет муниципалитет с помощью справочника
    '''

    mo_list = {'АЛЕКСЕЕВСКИЙ':'Алексеевский',
               'БЕЛГОРОД': 'Белгород',
               'БЕЛГОРОДСКИЙ РАЙОН': 'Белгородский район',
               'БОРИСОВСКИЙ': 'Борисовский',
               'ВАЛУЙСКИЙ': 'Валуйский',
               'ВЕЙДЕЛЕВСКИЙ': 'Вейделевский',
               'ВОЛОКОНОВСКИЙ': 'Волоконовский',
               'ГРАЙВОРОНСКИЙ': 'Грайворонский',
               'ГУБКИНСКИЙ': 'Губкинский',
               'ИВНЯНСКИЙ': 'Ивнянский',
               'КОРОЧАНСКИЙ': 'Корочанский',
               'КРАСНЕНСКИЙ': 'Красненский',
               'КРАСНОЯРУЖСКИЙ': 'Краснояружский',
               'КРАСНОГВАРДЕЙСКИЙ': 'Красногвардейский',
               'НОВООСКОЛЬСКИЙ': 'Новооскольский',
               'ПРОХОРОВСКИЙ': 'Прохоровский',
               'РАКИТЯНСКИЙ': 'Ракитянский',
               'РОВЕНЬСКИЙ': 'Ровеньский',
               'СТАРООСКОЛЬСКИЙ': 'Старооскольский',
               'ЧЕРНЯНСКИЙ': 'Чернянский',
               'ШЕБЕКИНСКИЙ': 'Шебекинский',
               'ЯКОВЛЕВСКИЙ': 'Яковлевский'}



    if (string is None):
        return None
    else:
        string = str(string).upper()
        patterns = spravochnik_street
        suff_nas_list = get_nas_punkt_new(string, return_list=True)
        nas_set = set([i[1] for i in suff_nas_list])
        suff_set = set([i[0] for i in suff_nas_list])
        pat_df = patterns.loc[
            (patterns["nas_punkt"].isin(nas_set)) & (patterns["nas_punkt_suffix"].isin(suff_set))].copy()
        # pat_df.loc[:, "full_nas"] = pat_df["nas_punkt_suffix"] + ". " + pat_df["nas_punkt"]
        pat_df["full_nas"] = pat_df["nas_punkt_suffix"] + ". " + pat_df["nas_punkt"]
        # print(pat_df["full_nas"])
        mo_res_list = []
        for mo, nas in zip(pat_df["mo"], pat_df["full_nas"]):
            street = get_street_new_opt(string, mo, nas)
            if street is not None:
                mo_res_list.append(mo)
        mo_set = set(mo_res_list)
        # print(mo_set)
        if len(mo_set) == 1:
            res = mo_list[list(mo_set)[0]]
            return res




# =================================== парсинг НОВЫЙ ПОДХОД ===============================
# загружаем справочник из БД
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

    # Выполнение SQL-запроса
    cur.execute(f'SELECT * FROM {name_table}')

    # Получение результатов
    result = cur.fetchall()
    result = pd.DataFrame(result)
    # Закрытие курсора и соединения
    cur.close()
    conn.close()
    return result


spravochnik_street = get_in_postgreSQL("public.spravochnik_street")
spravochnik_street.columns = ['mo', 'nas_punkt', 'street', 'nas_punkt_suffix', 'street_suffix']


# ========================================= ЗАГРУЖАЕМ ГАР =============================================

gar = pd.read_excel(r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address\gar.xlsx")



# ============================================ НАСЕЛЕННЫЙ ПУНКТ =======================================
def get_nas_punkt_new(string, mo=None, return_list=False):
    '''
    Определяет населенный пункт с помощью справочника
    '''

    if string is None:
        return None
    else:
        string = str(string).upper()
        nas_punkt_list = []
        patterns = spravochnik_street
        if mo is None:
            nas_punkt = patterns[["nas_punkt", "nas_punkt_suffix"]]
            nas_punkt = nas_punkt.drop_duplicates()
        else:
            mo = str(mo).upper()
            # print(mo)
            if mo != "БЕЛГОРОД":
                end_index = string.find(mo) + len(mo)
                string = string[end_index:]
            # print(string)
            nas_punkt = patterns.loc[(patterns["mo"] == mo)][["nas_punkt", "nas_punkt_suffix"]]
            nas_punkt = nas_punkt.drop_duplicates()
        # print(nas_punkt)
        for nas, suffix in zip(nas_punkt["nas_punkt"], nas_punkt["nas_punkt_suffix"]):
            # print(nas)
            match = re.search(fr"(\(|\s|,|\.|^){nas}(\s|,|$|\.|\))", string, re.IGNORECASE)
            if match:
                nas_punkt_list.append([suffix, nas])

        # print(nas_punkt_list)

        if return_list:  # возвращаем список найденных населенных пунктов
            return nas_punkt_list

        if len(nas_punkt_list) == 1:
            res = nas_punkt_list[0]
            return ". ".join(res)
        elif len(nas_punkt_list) > 1:
            set_suffix = set([i[0] for i in nas_punkt_list])
            # print(set_suffix)
            if len(set_suffix) > 1:
                for item in nas_punkt_list:
                    suffix = item[0]
                    # print(suffix)
                    suf_list = []
                    for suf in suffix_spr(suffix):
                        match_suffix = re.search(fr'''(\s|,){suf}(\s|\.)''', string, re.IGNORECASE)
                        if match_suffix:
                            suf_list.append([suffix, item[1]])
                    if len(suf_list) == 1:
                        return ". ".join(suf_list[0])
            else:
                set_nas = set([i[1] for i in nas_punkt_list])
                longest_word = max(set_nas, key=len)
                match = re.search(fr"(\(|\s|,|\.|^){longest_word}(\s|,|$|\.|\))", string, re.IGNORECASE)
                if match:
                    return list(set_suffix)[0] + '. ' + longest_word


        else:
            if mo is not None:
                return alter_nas(string, nas_punkt)


def alter_nas(string, nas_punkt):
    # print("alter_nas")
    komponent = ['Г', 'ГОРОД', 'ГОР', 'МКР',
                 'С', 'СЕЛО', 'П', 'ПОС', 'ПОСЕЛОК', 'ПОСЁЛОК', 'ПГТ',
                 'Х', 'ХУТОР', 'ДЕРЕВНЯ', 'СТ', 'СТАНЦИЯ']

    for nas in komponent:
        # print(nas)
        pat = fr"(\(|\s|,|\.|^){nas}(\s|,|$|\.|\))"

        match = re.search(pat, string)
        if match:
            # print("alter_nas " + pat)
            s = string[match.span()[1]:]

            sep_list = [" ", ",", "(", ")"]
            for sep in sep_list:
                ss = s.split(sep)
                # print(ss)
                for i in range(len(ss)):
                    r = sep.join(ss[:i]).strip(r"[ ,.]")
                    # print(r)

                    for nsp, suffix in zip(nas_punkt["nas_punkt"], nas_punkt["nas_punkt_suffix"]):
                        if r == nsp:
                            print("итог ", suffix, r)
                            return suffix + '. ' + r


def suffix_spr(suffix):
    # ['Г' 'С' 'П' 'Х' 'ПГТ' 'Г-К' 'РП' 'Д' 'СТ']
    spr = {'Г': ['Г', 'ГОРОД', 'ГОР'],
           'С': ['С', 'СЕЛО'],
           'П': ['П', 'ПОС', 'ПОСЕЛОК', 'ПОСЁЛОК', 'ПГТ'],
           'Х': ['Х', 'ХУТОР'],
           'ПГТ': ['ПГТ', 'П', 'ПОС', 'ПОСЕЛОК', 'ПОСЁЛОК'],
           'Г-К': ['Г-К'],
           'РП': ['РП'],
           'Д': ['Д', 'ДЕРЕВНЯ'],
           'СТ': ['СТ', 'СТАНЦИЯ']}
    for key, val in spr.items():
        if key == suffix:
            return val


# ===================================== УЛИЦА ====================================================

def get_street_new_opt(in_string, mo, nas_punkt):
    '''
    Определяет улицу с помощью справочника
    '''

    if (mo is None) | (nas_punkt is None) | (in_string is None):
        return None
    else:
        mo = mo.upper()
        nas_punkt = nas_punkt.upper()
        pref, nas_punkt = nas_punkt.split(maxsplit=1)
        pref = pref.strip(".")

        in_string = in_string.upper()

        match_mkr = re.search(fr'''(\s|,){"МКР"}(\s|\.|,)''', in_string, re.IGNORECASE)
        if match_mkr:
            res = if_mkr(in_string, mo, "МКР", pref, nas_punkt)
            return res

        match_cnt = re.search(fr'''(\s|,){"СНТ"}(\s|\.|,)''', in_string, re.IGNORECASE)
        if match_cnt:
            res = if_mkr(in_string, mo, "СНТ", pref, nas_punkt)
            return res

        # print(nas_punkt)
        end_index = in_string.find(fr"{nas_punkt}") + len(nas_punkt)
        string = in_string[end_index:]
        # print(string)
        suffix = None
        street_list = []
        suf_list = []
        patterns = spravochnik_street
        streets = spravochnik_street \
            .loc[(spravochnik_street["mo"] == mo) & (spravochnik_street["nas_punkt"] == nas_punkt)][
            ["street", "street_suffix"]]
        suffix_series = streets["street_suffix"].unique()
        # print(suffix_series)
        for suf in suffix_series:
            # print(suf)
            match_suffix = re.search(fr'''(\s|,){suf}(\s|\.|,)''', string, re.IGNORECASE)
            if match_suffix:
                # suffix = suf
                suf_list.append(suf)
                # print(suffix)
        # print(f"suffix: {suffix}")
        if len(suf_list) == 0:  # если суффикс не нашелся
            # print("суффиксов не нашлось")
            for street, suffix in zip(streets["street"], streets["street_suffix"]):
                # print(pat)
                match = re.search(fr"(\(|\s|,|\.|^){street}(\s|,|$|\.|\))", string, re.IGNORECASE)
                if match:
                    street_list.append([suffix, street])

                else:
                    if (" " in street) and ("-" in street):
                        street_revers = street.split(" ", 1)
                        street_revers = " ".join([street_revers[1], street_revers[0]])
                        match = re.search(fr"(\(|\s|,|\.|^){street_revers}(\s|,|$|\.|\))", string, re.IGNORECASE)
                        if match:
                            street_list.append([suffix, street])

            # print(street_list)
            if len(street_list) == 1:
                res = street_list[0]
                return ". ".join(res)

        elif len(suf_list) > 0:  # если нашлось больше 0
            # print("нашелся один суффикс " + suf_list[0])
            for suffix in suf_list:
                # suffix = suf_list[0]
                streets_suf = streets.loc[streets["street_suffix"] == suffix]["street"].unique()
                n = 0
                for street in streets_suf:
                    # print(pat)
                    match = re.search(fr"(\(|\s|,|\.|^){street}(\s|,|$|\.|\))", string, re.IGNORECASE)
                    if match:
                        return ". ".join([suffix, street])
                    else:
                        if (" " in street) and ("-" in street):
                            street_revers = street.split(" ", 1)
                            street_revers = " ".join([street_revers[1], street_revers[0]])
                            match = re.search(fr"(\(|\s|,|\.|^){street_revers}(\s|,|$|\.|\))", string, re.IGNORECASE)
                            if match:
                                return ". ".join([suffix, street])




def if_mkr(in_string, mo, type, pref, nas_punkt):
    # print("if_mkr")
    end_index = in_string.find(fr"{mo}") + len(mo)
    string = in_string[end_index:]
    # print(string)
    mk_list = []
    st_list = []

    type_list = []
    if type == "МКР":
        type_list = ["МКР", "МКР."]
    elif type == "СНТ":
        type_list = ["СНТ", "ТЕР. СНТ"]


    suffix = None
    match_suf = re.search(fr'''(\s|,){"УЛ"}(\s|\.|,)''', string, re.IGNORECASE)
    if match_suf:
        suffix = "УЛ"
    else:
        match_suf = re.search(fr'''(\s|,){"ПЕР"}(\s|\.|,)''', string, re.IGNORECASE)
        if match_suf:
            suffix = "ПЕР"

    spr_mo = gar.loc[gar.MO.notnull()][["OBJECTID", "MO"]]
    # print(mo)
    mun_id = spr_mo.loc[spr_mo.MO == mo].OBJECTID
    # print("mun_id", len(mun_id))
    if len(mun_id) > 0:
        mun_id = mun_id.values[0]
    else:
        return None

    nsp_id = gar.loc[(gar.TYPENAME == pref) & (gar[1] == int(mun_id)) & (gar.NAME == str(nas_punkt))].OBJECTID
    # print("nsp_id " + str(len(nsp_id)))
    if len(nsp_id) > 0:
        nsp_id = nsp_id.values[0]
    else:
        return None

    for i in range(5):
        mkr = gar.loc[(gar.TYPENAME.isin(type_list)) & (gar[1] == int(mun_id)) & (gar[i] == int(nsp_id))]
        if len(mkr) > 0:
            # print(mkr)
            break

    for id, mk in zip(mkr.OBJECTID, mkr.NAME):
        # print(mk)
        match = re.search(fr"(\(|\s|,|\.|^){mk}(\s|,|$|\.|\))", string, re.IGNORECASE)
        if match:
            mk_list.append([id, mk])
    # print(mk_list)
    if len(mk_list) > 0:
        for mk_id, mk in mk_list:
            if suffix is not None:

                for i in range(5):
                    street = gar.loc[(gar.TYPENAME == suffix) & (gar[i] == int(mk_id))]
                    if len(street) > 0:
                        # print(street)
                        break
                for id, st in zip(street.OBJECTID, street.NAME):
                    # print(st)
                    match = re.search(fr"(\(|\s|,|\.|^){st}(\s|,|$|\.|\))", string, re.IGNORECASE)
                    if match:
                        # print("сошлась улица")
                        st_list.append([id, st])
                # print(st_list)
            if len(st_list) == 1:
                return suffix + ". " + st_list[0][1]
        else:
            return type + ". " + mk_list[0][1]


# =================================================== ДОМ ===================================

def get_home_new(string, street):
    '''
        Определяет дом с помощью регулярного выражения
    '''

    if string is None:
        return None
    else:
        string = str(string).upper()
        if street is not None:
            street = street.split(maxsplit=1)[1].upper()
            end_index = string.find(street) + len(street)
            string = " " + string[end_index:]
            # print(string)

        levels = [
            r'(?<= Д | Д\.|,Д |,Д\.)\d{1,3}\s{0,1}[А-ЕИЖ]{0,1}([/-])?(?(1)(\d{1,3}(?=$|,)|[А-ЕИЖ]|(\d{1,3}\s{0,1}[А-ЕИЖ]))|([А-ЕИЖ]|$|\s|(?=,)))',
            r'(?<=.\s|.,|Д\.)\d{1,3}\s{0,1}[А-ЕИЖ]{0,1}([/-])?(?(1)(\d{1,3}(?=$|,)|[А-ЕИЖ]|(\d{1,3}\s{0,1}[А-ЕИЖ]))|([А-ЕИЖ]|$|\s|(?=,)))'
            # r'(?<=[\s,Д\.])\d{1,3}'
        ]
        for pattern in levels:
            match = re.search(pattern, string, flags=re.IGNORECASE)
            # print(match)
            if match:
                # print(match[0])
                return match.group().replace(" ", "")


# ========================================== КВАРТИРА ===================================================
def get_kvartira_new(string, street):
    '''
        Определяет дом с помощью регулярного выражения
    '''

    if string is None:
        return None
    else:
        string = str(string).upper()
        if street is not None:
            street = street.split(maxsplit=1)[1].upper()
            end_index = string.find(street) + len(street)
            string = " " + string[end_index:]
            # print(string)

        levels = [
            r"(?<=КВ\.|КВ | К )[ ]{0,1}\d+[А-я0-9/-]{0,5}(?=,|\s|$)"
        ]
        for pattern in levels:
            match = re.search(pattern, string, flags=re.IGNORECASE)
            # print(match)
            if match:
                # print(match[0])
                return match.group().strip(r"[/- ]")


# =======================================================================================================

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
