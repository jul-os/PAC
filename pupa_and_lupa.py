"""
## Лабораторная работа 3.1. ООП.
1. Реализовать два класса **Pupa** и **Lupa**. И класс **Accountant**.
2. Класс **Accountant** должен уметь одинаково успешно работать и с экземплярами класса
 **Pupa** и с экземплярами класса **Lupa**.
    У класса **Accountant** должен быть метод *give_salary(worker)*. Который, получая на
    вход экземпляр классов **Pupa** или **Lupa**, вызывает у них метод *take_salary(int)*.
    Необходимо придумать как реализовать такое поведение.
    Метод *take_salary* инкрементирует внутренний счётчик у каждого экземпляра класса на
    переданное ему значение.
3. При этом **Pupa** и **Lupa** два датасайнтиста и должны работать с матрицами.
У них есть метод *do_work(filename1, filename2)*.
    **Pupa** считывают из обоих переданных ему файлов по матрице и поэлементно
    их суммируют.
    **Lupa** считывают из обоих переданных ему файлов по матрице и поэлементно и
    х вычитают.
    Работники обоих типов выводят результат своих трудов на экран.

"""


class Accountant:

    def __init__(self, treasury=100, employers_count=2):
        self._treasury = treasury
        self._employers_count = employers_count

    def set_treasury(self, coin):
        self._treasury = coin

    def set_employers_count(self, employers_count):
        self._employers_count = employers_count

    def _how_much_salary(self):
        return self._treasury / self._employers_count

    def give_salary_to_Pupa(self, pupa):
        if isinstance(pupa, Pupa):
            coins_to_you = self._how_much_salary()
            pupa.take_salary(coin=coins_to_you)

    def give_salary_to_Lupa(self, lupa):
        if isinstance(lupa, Lupa):
            coins_to_you = self._how_much_salary()
            lupa.take_salary(coin=coins_to_you)


class Pupa:
    def __init__(self):
        self._salary = 0

    def take_salary(self, coin):  # public тк входит в интерфейс
        self._salary += coin

    def _read_matrix(self, _file):
        with open(_file, "r") as f:
            _matrix = [list(map(int, row.split())) for row in f.readlines()]
        return _matrix

    def _sum_matrix(self, _matrix1, _matrix2):
        if len(_matrix1) != len(_matrix2) or len(_matrix1[0]) != len(_matrix2[0]):
            print("I can't sum it!!")
            return 0
        _res = [[None for __ in range(len(_matrix1[0]))] for __ in range(len(_matrix1))]
        for i in range(len(_matrix1)):
            for j in range(len(_matrix1[0])):
                _res[i][j] = _matrix1[i][j] + _matrix2[i][j]
        return _res

    def _write_matrix(self, _matrix):
        return "\n".join(" ".join(list(map(str, row))) for row in _matrix)

    def do_work(self, filename1, filename2):
        _matrix1 = self._read_matrix(filename1)
        _matrix2 = self._read_matrix(filename2)
        _sum_matrix = self._sum_matrix(_matrix1, _matrix2)
        if _sum_matrix:
            print(self._write_matrix(_sum_matrix))


class Lupa:
    def __init__(self):
        self._salary = 0

    def take_salary(self, coin):  # public тк входит в интерфейс
        self._salary += coin

    def _read_matrix(self, _file):
        with open(_file, "r") as f:
            _matrix = [list(map(int, row.split())) for row in f.readlines()]
        return _matrix

    def _sub_matrix(self, _matrix1, _matrix2):
        if len(_matrix1) != len(_matrix2) or len(_matrix1[0]) != len(_matrix2[0]):
            print("I can't sum it!!")
            return 0
        _res = [[None for __ in range(len(_matrix1[0]))] for __ in range(len(_matrix1))]
        for i in range(len(_matrix1)):
            for j in range(len(_matrix1[0])):
                _res[i][j] = _matrix1[i][j] - _matrix2[i][j]
        return _res

    def _write_matrix(self, _matrix):
        return "\n".join(" ".join(list(map(str, row))) for row in _matrix)

    def do_work(self, filename1, filename2):
        _matrix1 = self._read_matrix(filename1)
        _matrix2 = self._read_matrix(filename2)
        _sub_matrix = self._sub_matrix(_matrix1, _matrix2)
        if _sub_matrix:
            print(self._write_matrix(_sub_matrix))


accountant = Accountant(treasury=200, employers_count=2)
pupa = Pupa()
lupa = Lupa()
pupa.do_work(filename1="matrix1.txt", filename2="matrix1.txt")
lupa.do_work(filename1="matrix1.txt", filename2="matrix1.txt")
accountant.give_salary_to_Pupa(lupa)
accountant.give_salary_to_Lupa(pupa)
print(lupa._salary)
print(pupa._salary)
