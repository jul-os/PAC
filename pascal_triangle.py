"""
### Лабораторная работа 1.2. Треугольник Паскаля

Реализовать скрипт pascal_triangle.py

Входные параметры скрипта: высота треугольника = количество строк в нём
Выходные значения: выводит треугольник Паскаля на экран

Что должен делать скрипт:
1) Распарсить входные параметры скрипта с помощью [argparse](https://docs.python.org/3/library/argparse.html)
2) Построить [треугольник Паскаля](https://ru.wikipedia.org/wiki/%D0%A2%D1%80%D0%B5%D1%83%D0%B3%D0%BE%D0%BB%D1%8C%D0%BD%D0%B8%D0%BA_%D0%9F%D0%B0%D1%81%D0%BA%D0%B0%D0%BB%D1%8F)
из N строк (подаётся в качестве входного значения скрипта, получается в пункте 1)
3) Вывести треугольник на экран. Обратите внимание, что треугольник должен выводиться
в формате равнобедренного треугольника. Для этого в начале каждой строки необходимо добавить
нужное число пробелов.
"""

import argparse


def print_pascal_triangle(P):
    n = len(P)
    for i, row in enumerate(P):
        spaces = " " * (n - i - 1)
        numbers_str = " ".join(str(num) for num in row)
        print(spaces + numbers_str)


def make_pascal_triangle(N, P):
    for i in range(0, N):
        row = [1] * (i + 1)
        for j in range(i + 1):
            if j != 0 and j != i:
                row[j] = P[i - 1][j - 1] + P[i - 1][j]
        P.append(row)
    return P


parser = argparse.ArgumentParser()
parser.add_argument("number", type=int, help="Целое число")
args = parser.parse_args()
N = args.number
P = []
P = make_pascal_triangle(N, P)
print_pascal_triangle(P)
