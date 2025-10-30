import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# копию дома для отрисовки результатов
result_img = house.copy()

# Initiate SIFT detector
sift = cv2.SIFT_create()


# Функция для поиска всех призраков одного типа с масками
def find_all_ghosts(house_img, ghost_img, ghost_name, max_attempts=10):
    if ghost_name == "scary_ghost":
        MIN_MATCH_COUNT = 2
    else:
        MIN_MATCH_COUNT = 10
    house_working = house_img.copy()
    all_ghosts_found = []

    for attempt in range(max_attempts):
        kp_house, des_house = sift.detectAndCompute(house_working, None)
        kp_ghost, des_ghost = sift.detectAndCompute(ghost_img, None)

        if (
            des_house is None
            or des_ghost is None
            or len(des_house) < MIN_MATCH_COUNT
            or len(des_ghost) < MIN_MATCH_COUNT
        ):
            break

        # FLANN параметры
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Сопоставляем дескрипторы
        matches = flann.knnMatch(des_ghost, des_house, k=2)

        # Фильтруем хорошие matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        print(
            f"{ghost_name} - попытка {attempt+1}: найдено {len(good_matches)} хороших совпадений"
        )

        if len(good_matches) > MIN_MATCH_COUNT:
            # Получаем координаты точек для homography
            src_pts = np.float32(
                [kp_ghost[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_house[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # Находим homography с RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Получаем координаты рамки призрака
                h, w = ghost_img.shape[
                    :2
                ]  # Получаем высоту и ширину изображения призрака
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(
                    -1, 1, 2
                )  # берем углы этого изображения и решэйпим
                dst = cv2.perspectiveTransform(pts, M)
                """Матрица M - это homography матрица, которую мы получили из 
                cv2.findHomography(). Она описывает преобразование, которое переводит точки 
                из системы координат призрака в систему координат дома.Homography матрица 
                (3×3) делает аффинное преобразование + перспективные искажения:"""

                # Проверяем, что найденная область разумного размера
                dst_int = np.int32(dst)
                x_coords = dst_int[:, 0, 0]
                y_coords = dst_int[:, 0, 1]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)

                if (
                    5 < width < house_working.shape[1]
                    and 5 < height < house_working.shape[0]
                ):
                    all_ghosts_found.append(dst_int)

                    # маскируем найденную область
                    cv2.fillPoly(house_working, [dst_int], (0, 0, 0))

                    print(f"  Найден {ghost_name} #{len(all_ghosts_found)}")
                else:
                    print(f"  Пропуск {ghost_name} - неправильный размер")
                    break
            else:
                print(f"  Homography не найдена для {ghost_name}")
                break
        else:
            print(f"  Недостаточно совпадений для {ghost_name}")
            break

    return all_ghosts_found


ghost_colors = {
    "candy_ghost": (0, 255, 0),
    "pampkin_ghost": (0, 255, 255),
    "scary_ghost": (255, 0, 0),
}

print("Начинаем поиск candy_ghost...")
candy_ghosts = find_all_ghosts(house, candy_ghost, "candy_ghost")

print("\nНачинаем поиск pampkin_ghost...")
pampkin_ghosts = find_all_ghosts(house, pampkin_ghost, "pampkin_ghost")

print("\nНачинаем поиск scary_ghost...")
scary_ghosts = find_all_ghosts(house, scary_ghost, "scary_ghost")
mirrored_scary_ghosts = find_all_ghosts(
    house, cv2.flip(scary_ghost, 1), "mirrored_scary_ghost"
)

total_ghosts = 0

for i, ghost_box in enumerate(candy_ghosts):
    result_img = cv2.polylines(
        result_img, [ghost_box], True, ghost_colors["candy_ghost"], 3, cv2.LINE_AA
    )
    center = np.mean(ghost_box, axis=0).astype(int)[0]
    cv2.putText(
        result_img,
        f"C{i+1}",
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        ghost_colors["candy_ghost"],
        2,
    )
    total_ghosts += 1

for i, ghost_box in enumerate(pampkin_ghosts):
    result_img = cv2.polylines(
        result_img, [ghost_box], True, ghost_colors["pampkin_ghost"], 3, cv2.LINE_AA
    )
    center = np.mean(ghost_box, axis=0).astype(int)[0]
    cv2.putText(
        result_img,
        f"P{i+1}",
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        ghost_colors["pampkin_ghost"],
        2,
    )
    total_ghosts += 1

for i, ghost_box in enumerate(scary_ghosts):
    result_img = cv2.polylines(
        result_img, [ghost_box], True, ghost_colors["scary_ghost"], 3, cv2.LINE_AA
    )
    center = np.mean(ghost_box, axis=0).astype(int)[0]
    cv2.putText(
        result_img,
        f"S{i+1}",
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        ghost_colors["scary_ghost"],
        2,
    )
    total_ghosts += 1
for i, ghost_box in enumerate(mirrored_scary_ghosts):
    result_img = cv2.polylines(
        result_img, [ghost_box], True, ghost_colors["scary_ghost"], 3, cv2.LINE_AA
    )
    center = np.mean(ghost_box, axis=0).astype(int)[0]
    cv2.putText(
        result_img,
        f"S{i+1}",
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        ghost_colors["scary_ghost"],
        2,
    )
    total_ghosts += 1
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f"Призраки!: {total_ghosts}")
plt.axis("off")
plt.show()

cv2.imwrite("ghost_detection_result.jpg", result_img)
print("Результат сохранен как 'ghost_detection_result.jpg'")
