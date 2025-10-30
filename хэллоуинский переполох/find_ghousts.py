import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

house = cv2.imread(
    "/home/julia/Рабочий стол/Новая папка/PAC/хэллоуинский переполох/lab7.png"
)
candy_ghost = cv2.imread(
    "/home/julia/Рабочий стол/Новая папка/PAC/хэллоуинский переполох/candy_ghost.png"
)
pampkin_ghost = cv2.imread(
    "/home/julia/Рабочий стол/Новая папка/PAC/хэллоуинский переполох/pampkin_ghost.png"
)
scary_ghost = cv2.imread(
    "/home/julia/Рабочий стол/Новая папка/PAC/хэллоуинский переполох/scary_ghost.png"
)
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp0, des0 = sift.detectAndCompute(house, None)
kp1, des1 = sift.detectAndCompute(candy_ghost, None)
kp2, des2 = sift.detectAndCompute(pampkin_ghost, None)
kp3, des3 = sift.detectAndCompute(scary_ghost, None)


# FLANN параметры для SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # или больше для большей точности

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Сопоставляем дескрипторы дома и candy_ghost
matches = flann.knnMatch(des1, des0, k=2)  # des1 - призрак, des0 - дом
# Сохраняем хорошие matches согласно тесту Лоу
good_matches1 = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # классическое значение 0.7
        good_matches1.append(m)
if len(good_matches1) > MIN_MATCH_COUNT:
    # Получаем координаты точек для homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp0[m.trainIdx].pt for m in good_matches1]).reshape(-1, 1, 2)

    # Находим homography с RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Получаем координаты рамки призрака
    h, w = candy_ghost.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Рисуем рамку на house
    house_with_box = cv2.polylines(
        house, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA
    )
    cv2.imshow("res", house_with_box)
    cv2.waitKey(0)

    print("candy_ghost найден и обведен!")
else:
    print(f"candy_ghost не найден. Мало соответствий: {len(good_matches1)}")


cv2.destroyAllWindows()
