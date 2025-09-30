"""
Входные параметры скрипта:
1. Путь к файлу с матрицами. Например, matrix.txt. Внутри файла заданы две целочисленные матрицы.
2. Путь к файлу с результатом работы программы

Выходные значения: записывает результирующую матрицу в файл с результатом работы программы

Что должен делать скрипт:
1) Прочитать файл, используя встроенные функции Python для работы с файлами.
2) Преобразовать прочитанные строки в матрицы. Матрицы реализовать используя стандартные типы данных Python. Например, список списков.
3) Найти произведение полученных матриц.
4) Записать результат произведения в выходной файл.
"""

import argparse
import sys
import numpy as np


def print_matrix(matrix):
    return "\n".join(" ".join(list(map(str, row))) for row in matrix)


def open_files():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="your input file"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="your input file"
    )
    args = parser.parse_args()
    if not args.input_file or not args.output_file:
        parser.print_usage()
        sys.exit(EXIT_FAILURE)
    input = args.input_file
    output = args.output_file
    return input, output


def matrix_init(input):
    matrix = [list(map(int, row.split())) for row in input.readlines()]
    for i, row in enumerate(matrix):
        if row == []:
            matrix2 = matrix[(i + 1) :]
            matrix = matrix[:i]
    print("matrix A:", print_matrix(matrix), sep="\n")
    print("matrix B:", print_matrix(matrix2), sep="\n")
    return matrix, matrix2


def matrix_mult(matrix1, matrix2):
    m = len(matrix1)  # matrix1: m × n
    n1 = len(matrix1[0])
    n = len(matrix2)  # matrix2: n × k
    if m == 1 and n1 == 1:
        res = matrix2
        for i in range(n):
            for j in range(len(matrix2[0])):
                res[i][j] = matrix2[i][j] * matrix1[0][0]
        return res
    elif n == 1 and len(matrix2[0]) == 1:
        res = matrix1
        for i in range(m):
            for j in range(n1):
                res[i][j] = matrix1[i][j] * matrix2[0][0]
        return res

    if n1 != n:
        print("cannot be multiplied!\n")
        return 0
    k = len(matrix2[0])

    res = [[None for __ in range(k)] for __ in range(m)]  # res: m × k

    for i in range(m):
        for j in range(k):
            res[i][j] = sum(matrix1[i][kk] * matrix2[kk][j] for kk in range(n))
    print("result matrix:", print_matrix(res), sep="\n")
    return res


def convolve(matrix, kernel):
    matrix_np = np.array(matrix)
    kernel_np = np.array(kernel)

    kr, kc = kernel_np.shape
    mr, mc = matrix_np.shape

    result_rows = mr - kr + 1
    result_cols = mc - kc + 1

    res = np.empty((result_rows, result_cols))

    for row in range(result_rows):
        for col in range(result_cols):
            res[row][col] = (
                matrix_np[row : row + kr, col : col + kc] * kernel_np
            ).sum()

    return res.tolist()


input, output = open_files()
matrix1, matrix2 = matrix_init(input)
mult = matrix_mult(matrix1, matrix2)
if mult:
    output.write(print_matrix(mult))

# свертки
"""
3) Найти свёртку первой матрицы со второй. Вторую матрицу использовать как ядро свёртки.
 Не нужно добавлять падинг, не нужно использовать шаг ядра свёртки  != 1.  [Wiki](http://neerc.ifmo.ru/wiki/index.php?title=%D0%A1%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D0%B5_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D0%B5_%D1%81%D0%B5%D1%82%D0%B8&mobileaction=toggle_view_desktop#.D0.A1.D0.B2.D0.B5.D1.80.D1.82.D0.BE.D1.87.D0.BD.D1.8B.D0.B9_.D1.81.D0.BB.D0.BE.D0.B9) [YouTube](https://youtu.be/vcFPxZMVurQ)
4) Записать результат произведения в выходной файл.
"""

# мне лень поэтому пусть вторая матрица всегда ядро
convol = convolve(matrix1, matrix2)
output.write("Свертка")
print("свертка\n", print_matrix(convol))
output.write(print_matrix(convol))
