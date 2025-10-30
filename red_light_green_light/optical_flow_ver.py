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
import time
import random


class RedLightGreenLight:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            exit()
        self.orig_doll = cv2.imread("images.jpg")
        self.turned_away_doll = cv2.imread("turned.png")
        self.over = cv2.imread("game over.jpg")
        self.win = cv2.imread("win.jpg")
        if self.orig_doll is None or self.turned_away_doll is None:
            print("Couln't open images")
            exit()
        self.is_red_light = False
        self.mode_start_time = time.time()
        self.curr_mode_duration = random.randint(3, 10)  # от 3 до 10 секунд
        self.cycle_count = 0
        # params for corner detection
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.prev_points = None
        self.track_len = 10
        self.tracks = []

    def check_mode_switch(self):
        curr_time = time.time()
        if curr_time - self.mode_start_time > self.curr_mode_duration:
            self.is_red_light = not self.is_red_light
            self.mode_start_time = curr_time
            self.curr_mode_duration = random.randint(3, 10)
            self.cycle_count += 1
            # Reset tracking when mode switches
            self.prev_points = None
            return True
        return False

    def red_light_mode(self, frame, prev_gray):
        game_over = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        doll = cv2.resize(self.orig_doll, (frame.shape[1], frame.shape[0]))
        doll = cv2.addWeighted(
            doll, 0.7, np.full(doll.shape, (0, 0, 255), dtype=np.uint8), 0.3, 0
        )
        if prev_gray is not None:
            # If we don't have points to track, detect new ones
            if self.prev_points is None or len(self.prev_points) < 10:
                self.prev_points = cv2.goodFeaturesToTrack(
                    prev_gray, mask=None, **self.feature_params
                )
            if self.prev_points is not None and len(self.prev_points) > 0:
                # Calculate optical flow
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, self.prev_points, None, **self.lk_params
                )
                # Select good points
                if next_points is not None:
                    good_old = self.prev_points[status == 1]
                    good_new = next_points[status == 1]

                    # Calculate movement magnitude
                    if len(good_old) > 0 and len(good_new) > 0:
                        movement_magnitude = np.sqrt(
                            np.sum((good_new - good_old) ** 2, axis=1)
                        )
                        avg_movement = np.mean(movement_magnitude)

            if avg_movement > 2.0:
                cv2.putText(
                    frame,
                    "I SAW YOU MOVE",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                game_over = True

            # Update previous points for next frame
            self.prev_points = good_new.reshape(-1, 1, 2)
        return frame, doll, gray, game_over

    def green_light_mode(self, frame):
        doll = cv2.resize(self.turned_away_doll, (frame.shape[1], frame.shape[0]))
        doll = cv2.addWeighted(
            doll, 0.9, np.full(doll.shape, (0, 255, 0), dtype=np.uint8), 0.3, 0
        )
        return frame, doll

    def run(self):
        prev_frame = None
        game_over = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Не удалось получить кадр")
                break
            self.check_mode_switch()
            if self.is_red_light:
                frame, doll, gray, game_over = self.red_light_mode(frame, prev_frame)
                prev_frame = gray  # сохраняем grayscale для следующего вызова

            else:
                frame, doll = self.green_light_mode(frame)
                prev_frame = None

            cv2.putText(
                frame,
                str(self.cycle_count),
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            combined = np.hstack((frame, doll))
            if game_over:
                # combined = np.hstack((frame, doll))
                start_time = time.time()
                cv2.imshow("Game :)", combined)
                while time.time() - start_time < 2:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                cv2.imshow("Game :)", self.over)
                start_time = time.time()
                while time.time() - start_time < 2:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                break
            cv2.imshow("Game :)", combined)
            if self.cycle_count == 5:
                cv2.imshow("Game :)", self.win)
                start_time = time.time()
                while time.time() - start_time < 2:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                break
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = RedLightGreenLight()
    game.run()


# todo плотный optical flow
