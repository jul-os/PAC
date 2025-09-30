"""### Лабораторная работа 1.1. Пузырьковая сортировка.

Реализовать скрипт bubble_sort.py

Входные параметры скрипта: целое число = длина списка.
Выходные значения: выводит отсортированный список на экран

Что должен делать скрипт:
1) Распарсить входные параметры скрипта с помощью [argparse](https://docs.python.org/3/library/argparse.html)
2) Создать список случайных значений от 0 до 1 длины N (подаётся в качестве входного значения скрипта, получается в пункте 1)
3) Реализовать пузырьковую сортировку с помощью операторов for и if. Никаких sorted!
4) Вывести значения на экран
"""

# python bubble_sort.py 3

import argparse
import random


def bubble_sort(nums, N):
    for i in range(N):
        swapped = False
        for j in range(0, N - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                swapped = True
        if not swapped:
            break
    return nums


parser = argparse.ArgumentParser()
parser.add_argument("number", type=int, help="Целое число")
args = parser.parse_args()
N = args.number
nums = [random.random() for i in range(N)]
print(nums)
nums = bubble_sort(nums, N)
print(nums)
