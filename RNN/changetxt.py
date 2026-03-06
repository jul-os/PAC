import re

# Имя входного и выходного файлов
input_file = '/home/julia/Рабочий стол/code/PAC/RNN/Филип Кинред Дик - Мечтают ли андроиды об электроовцах.txt'
output_file = '/home/julia/Рабочий стол/code/PAC/RNN/измененные андроиды.txt'

# Чтение файла
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Удаление всего, кроме букв (латиница+кириллица) и пробелов
# ^ внутри [] означает "все, кроме"
# \s - включает пробелы, табуляцию и переносы строк
cleaned_content = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', content)

# Запись результата
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print(f"Файл очищен. Результат сохранен в {output_file}")
