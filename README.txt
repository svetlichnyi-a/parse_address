Проект по парсингу строки с адресом.
Сделан, чтобы применять в других проектах.
Чтобы его импортировать в другой проект нужно написать:

import sys
PATH = r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address"
sys.path.insert(0, PATH)
from parse_address import *

либо указать конкретные функции

import sys
import pandas as pd
PATH = r"C:\Users\svetlichnyy_av\PycharmProjects\parse_address"
sys.path.insert(0, PATH)
from parse_address import get_mo_re, get_mo_neiroset, parse_address