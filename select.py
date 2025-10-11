# Выбор случайных элементов массива
"""
Есть два набора данных: реальные и синтетические. Допустим, мы хотим обучить некоторую ML модель на смеси
реальных и синтетических данных. При этом синтетические данные должны браться с вероятностью P.
Важно сохранять порядок входных чисел.
Например: Для массивов: [1,2,3,4,5,7,8,9,10] и [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] и P=0.2
Один из вариантов возвращаемого значения: [1,-2,3,4,-5,6,7,8,9,10]

Массивы реальных и синтетических данных одинаковой длины.

Реализовать скрипт random_select.py
Входные параметры скрипта: пути к двум файлам со списком целых чисел в каждом и вероятность P от 0 до 1.
Например python2 random_select.py file_1.txt file_2.txt 0.3
    file_1.txt содержит:
    1 2 3 4 5 6 7
    а file_2.txt
    -1 -2 -3 -4 -5 -6 -7
Результат перемешивания массивов вывести на экран.
Перемешивание масивов должно происходить тремя РАЗЛИЧНЫМИ способами.
"""
import argparse
import sys
import random
import numpy as np


def parse_aarguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input1", type=argparse.FileType("r"), help="first input file")
    parser.add_argument(
        "input2", type=argparse.FileType("r"), help="fisecondrst input file"
    )
    parser.add_argument("P", type=float, help="probability")
    args = parser.parse_args()
    if not args.input1 or not args.input2:
        parser.print_usage()
        sys.exit(EXIT_FAILURE)
    in1 = args.input1
    in2 = args.input2
    prob = args.P
    return in1, in2, prob


def read_data(file_obj):
    data = file_obj.read().strip().split()
    return [int(x) for x in data]


def method1(list1, list2, prob):
    result = []
    num_synth = int(len(list2) * prob)
    # Создаем массив индексов, перемешиваем, первые num будут добавлены
    ind = list(range(len(list1)))
    random.shuffle(ind)
    synth_ind = set(ind[:num_synth])
    for i in range(len(list2)):
        result.append(list2[i]) if i in synth_ind else result.append(list1[i])
    return result


def method2(list1, list2, p):
    n = len(list1)
    mask1 = np.random.rand(n) < p
    return np.where(mask1, list2, list1)


def method3(list1, list2, p):
    choices = np.random.choice([0, 1], size=len(list1), p=[1 - p, p])
    return np.where(choices, list2, list1)


def method4(list1, list2, p):
    mask3 = np.random.uniform(0, 1, len(list1)) < p
    return np.select([mask3], [list2], default=list1)


in1, in2, prob = parse_aarguments()
list1 = read_data(in1)
list2 = read_data(in2)
print(method1(list1, list2, prob))
print(method2(list1, list2, prob))
print(method3(list1, list2, prob))
print(method4(list1, list2, prob))
