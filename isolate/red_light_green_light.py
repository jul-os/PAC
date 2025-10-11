"""
Программа должна реализовывать следующий функционал:

1. ! Покадровое получение видеопотока с камеры. Использовать камеру ноутбука,
вебкамеру или записать видео файл с вебкамеры товарища и использовать его.
2.  ! Реализовать обнаружение движения в видеопотоке: попарно сравнивать текущий и
предыдущий кадры.
3. ! По мере проигрывания видео в отдельном окне отрисовывать двухцветную карту с
результатом: красное - есть движение, зелёное - нет движения
4. Добавить таймер, по которому включается и выключается обнаружение движения.
О текущем режиме программы сообщать текстом с краю изображения: “Красный свет” -
движение обнаруживается, “Зелёный свет” - движение не обнаруживается.
5. Реализовать более сложный алгоритм обнаружения движения, устойчивый к шумам
вебкамеры (OpticalFlow)
"""

import cv2
import numpy as np

orig_doll = cv2.imread("/home/julia/Рабочий стол/Новая папка/PAC/isolate/images.jpg")
turned_away_doll = cv2.imread(
    "/home/julia/Рабочий стол/Новая папка/PAC/isolate/turned.png"
)
# Получение видео. Объект VideoCapture позволяет работать с видеопотоком с камеры или из файла
# Для работы с файлом указывается путь к нему, для работы с вебкамерой указывается её номер, начиная с 0
cap = cv2.VideoCapture(0)

# Проверить, открылась ли камера
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()
prev_frame = None
while True:
    ret, frame = cap.read()
    orig_doll = cv2.resize(orig_doll, (frame.shape[1], frame.shape[0]))
    doll = orig_doll.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changes = np.sum(thresh > 0)
        if changes > 150:
            cv2.putText(
                frame,
                "I SAW YOU MOVE",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            doll = cv2.addWeighted(
                doll, 0.7, np.full(doll.shape, (0, 0, 255), dtype=np.uint8), 0.3, 0
            )
        else:
            doll = cv2.addWeighted(
                doll, 0.9, np.full(doll.shape, (0, 255, 0), dtype=np.uint8), 0.3, 0
            )
    prev_frame = gray.copy()
    combined = np.hstack((frame, doll))

    key = cv2.waitKey(20) & 0xFF
    cv2.imshow("Game :)", combined)
    if key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows
