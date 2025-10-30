import pandas as pd
import numpy as np

df1 = pd.read_csv(
    "/home/julia/Рабочий стол/Новая папка/PAC/data/lesson5/cinema_sessions.csv", sep=" "
)
df2 = pd.read_csv(
    "/home/julia/Рабочий стол/Новая папка/PAC/data/lesson5/titanic_with_labels.csv",
    sep=" ",
)
print(df1.head())

"""
Данные, которые предоставил кинотеатр находятся в файлах *cinema_sessions.csv* и
 *titanic_with_labels*  
  
1. Пол (sex): отфильтровать строки, где пол не указан, преобразовать оставшиеся в число 0/1;  
2. Номер ряда в зале (row_number): заполнить вместо NAN максимальным значением ряда;  
3. Количество выпитого в литрах (liters_drunk): отфильтровать отрицательные значения и 
нереально большие значения (выбросы). Вместо них заполнить средним.  
"""
df2 = df2[df2["sex"].notna()]  # удаляем строки с NaN
df2 = df2[df2["sex"] != ""]  # удаляем строки с пустыми строками

df2["sex"] = (
    df2["sex"]
    .str.lower()
    .replace({"м": 1, "m": 1, "M": 1, "М": 1, "ж": 0, "f": 0, "F": 0, "Ж": 0})
)

df2["row_number"] = df2["row_number"].fillna(df2["row_number"].max())

Q1 = df2["liters_drunk"].quantile(0.25)
Q3 = df2["liters_drunk"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

lower_bound = max(lower_bound, 0)

mean_normal = df2[
    (df2["liters_drunk"] >= lower_bound) & (df2["liters_drunk"] <= upper_bound)
]["liters_drunk"].mean()

df2["liters_drunk"] = np.where(
    (df2["liters_drunk"] >= lower_bound) & (df2["liters_drunk"] <= upper_bound),
    df2["liters_drunk"],
    mean_normal,
)


"""
## Лабораторная 7.2
4. Возраст (age): разделить на 3 группы: дети (до 18 лет), взрослые (18 - 50), 
пожилые (50+). закодировать в виде трёх столбцов с префиксом age_. Старый столбец с age удалить;
5. Напиток (drink): преобразовать в число 0/1 был ли этот напиток хмельным;
6. Номер чека (check_number): надо сопоставить со второй таблицей со временем сеанса. 
И закодировать в виде трёх столбцов, был ли это утренний (morining) сеанс, дневной (day) 
или вечерний (evening).
"""

df2["age_children"] = (df2["age"] < 18).astype(int)
df2["age_adults"] = ((df2["age"] >= 18) & (df2["age"] < 50)).astype(int)
df2["age_elderly"] = (df2["age"] >= 50).astype(int)

df2 = df2.drop("age", axis=1)


alchohol_drinks = ["Beerbeer", "Bugbeer", "Strong beer", "Наше пиво"]
df2["drink"] = np.where(df2["drink"].isin(alchohol_drinks), 1, 0)

print("\n\n", df2.head())


merged_df = df2.merge(
    df1[["check_number", "session_start"]], on="check_number", how="left"
)
merged_df["session_start"] = pd.to_datetime(merged_df["session_start"])

merged_df["hour"] = merged_df["session_start"].dt.hour


def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "day"
    else:
        return "evening"


merged_df["time_of_day"] = merged_df["hour"].apply(get_time_of_day)

time_dummies = pd.get_dummies(merged_df["time_of_day"], prefix="session")

merged_df = pd.concat([merged_df, time_dummies], axis=1)

merged_df = merged_df.drop(["session_start", "hour", "time_of_day"], axis=1)

print(merged_df.head())
