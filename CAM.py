"""
Исправленная версия CAM с веб-камеры
"""

import cv2
import torch
import numpy as np
from torchvision import models, transforms
import json
import urllib.request
from PIL import Image
import sys

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

print("Загрузка модели...")

# Исправленный способ загрузки модели (без предупреждений)
try:
    # Для новых версий PyTorch
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
except:
    # Для старых версий
    net = models.resnet50(pretrained=True)

net.eval()

# Метки классов
try:
    url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    with urllib.request.urlopen(url) as response:
        classes = [item[1] for item in json.loads(response.read().decode()).values()]
    print(f"Загружено {len(classes)} классов")
except Exception as e:
    print(f"Ошибка загрузки меток: {e}")
    # Запасной вариант - базовые классы
    classes = [f"Class #{i}" for i in range(1000)]

# Хук для признаков
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.detach().cpu().numpy())

# Регистрируем хук
hook_attached = False
for name, module in net.named_modules():
    if name == "layer4":
        module.register_forward_hook(hook_feature)
        print(f"Хук зарегистрирован на {name}")
        hook_attached = True
        break

if not hook_attached:
    print("ВНИМАНИЕ: Хук не зарегистрирован!")

# Веса последнего слоя
try:
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].detach().cpu().numpy())
    print(f"Размерность весов: {weight_softmax.shape}")
except Exception as e:
    print(f"Ошибка получения весов: {e}")
    weight_softmax = np.zeros((1000, 2048))

# Предобработка
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== ФУНКЦИИ ====================

def get_cam_and_prediction(frame):
    global features_blobs
    features_blobs = []  # Очищаем перед каждым кадром
    
    try:
        # Подготовка кадра
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(img_pil).unsqueeze(0)
        
        # Предсказание
        with torch.no_grad():
            logit = net(img_tensor)
        
        # Вероятности
        probs = torch.nn.functional.softmax(logit, dim=1)[0]
        top_prob, top_idx = probs.topk(1)
        top_prob = top_prob.item()
        top_idx = top_idx.item()
        
        # CAM
        cam = None
        if features_blobs and len(features_blobs) > 0:
            cam = returnCAM_simple(features_blobs[0], weight_softmax, top_idx)
        
        return cam, top_idx, top_prob
    except Exception as e:
        print(f"Ошибка обработки кадра: {e}")
        return None, 0, 0.0

def returnCAM_simple(feature_conv, weight_softmax, class_idx):
    """Упрощенная версия returnCAM для одного класса"""
    try:
        nc, h, w = feature_conv.shape[1:]
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        
        # Нормализация с защитой от деления на ноль
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return np.uint8(255 * cam)
    except Exception as e:
        print(f"Ошибка CAM: {e}")
        return np.zeros((7, 7), dtype=np.uint8)

# ==================== ОСНОВНОЙ ЦИКЛ ====================

def main():
    # Пробуем разные индексы камеры
    cap = None
    for camera_id in [0, 1, -1]:
        print(f"Пробуем открыть камеру {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"Камера {camera_id} открыта")
            break
    
    if cap is None or not cap.isOpened():
        print("Не удалось открыть камеру")
        return
    
    # Настройки камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*50)
    print("УПРАВЛЕНИЕ:")
    print("  Q или ESC - выход")
    print("  S - сохранить кадр")
    print("  F - показать/скрыть CAM")
    print("="*50 + "\n")
    
    show_cam = True
    frame_count = 0
    
    while True:
        # Захватываем кадр
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break
        
        # Обрабатываем каждый 3-й кадр для производительности
        process_this = (frame_count % 3 == 0)
        
        if process_this:
            cam, class_idx, prob = get_cam_and_prediction(frame)
            last_good_result = (cam, class_idx, prob)
        else:
            # Используем последний хороший результат
            cam, class_idx, prob = last_good_result if 'last_good_result' in locals() else (None, 0, 0.0)
        
        # Создаем результат
        if show_cam and cam is not None:
            # Масштабируем CAM до размера кадра
            cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            result = cv2.addWeighted(heatmap, 0.3, frame, 0.7, 0)
        else:
            result = frame.copy()
        
        # Добавляем текст с классом
        class_name = classes[class_idx] if class_idx < len(classes) else f"Class #{class_idx}"
        
        # Рисуем фон для текста
        cv2.rectangle(result, (5, 5), (400, 80), (0, 0, 0), -1)
        
        # Основной текст
        text = f"{class_name}"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 0), 2)
        
        # Вероятность
        prob_text = f"Confidence: {prob:.1%}"
        color = (0, 255, 0) if prob > 0.5 else (0, 255, 255) if prob > 0.2 else (0, 0, 255)
        cv2.putText(result, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        # Информация о режиме
        mode_text = "CAM: ON" if show_cam else "CAM: OFF"
        cv2.putText(result, mode_text, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Показываем результат
        cv2.imshow('CAM with Webcam', result)
        
        # Обработка клавиш - ИСПРАВЛЕНО!
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 27 = ESC
            print("Выход по запросу")
            break
        elif key == ord('s'):
            # Сохраняем кадр
            filename = f"cam_capture_{frame_count}.jpg"
            cv2.imwrite(filename, result)
            print(f"Кадр сохранён как {filename}")
        elif key == ord('f'):
            show_cam = not show_cam
            print(f"CAM: {'включен' if show_cam else 'выключен'}")
        
        frame_count += 1
    
    # Освобождаем ресурсы
    print("Освобождение ресурсов...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Дополнительно закрываем все окна OpenCV
    for i in range(5):
        cv2.waitKey(1)
    """
Простая версия CAM с веб-камеры
"""

import cv2
import torch
import numpy as np
from torchvision import models, transforms
import json
import urllib.request
from PIL import Image

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

print("Загрузка модели...")
# Метки классов
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    classes = [item[1] for item in json.loads(response.read().decode()).values()]

# Модель
net = models.resnet50(pretrained=True)
net.eval()

# Хук для признаков
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.detach().cpu().numpy())

for name, module in net.named_modules():
    if name == "layer4":
        module.register_forward_hook(hook_feature)
        break

# Веса последнего слоя
weight_softmax = np.squeeze(list(net.parameters())[-2].detach().cpu().numpy())

# Предобработка
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== ФУНКЦИИ ====================

def get_cam_and_prediction(frame):
    global features_blobs
    features_blobs = []
    
    # Подготовка кадра
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img_pil).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        logit = net(img_tensor)
    
    # Вероятности
    probs = torch.nn.functional.softmax(logit, dim=1)[0]
    top_prob, top_idx = probs.topk(1)
    
    # CAM
    if features_blobs:
        cam = returnCAM_simple(features_blobs[0], weight_softmax, top_idx.item())
        return cam, top_idx.item(), top_prob.item()
    return None, top_idx.item(), top_prob.item()

def returnCAM_simple(feature_conv, weight_softmax, class_idx):
    """Упрощенная версия returnCAM для одного класса"""
    nc, h, w = feature_conv.shape[1:]
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return np.uint8(255 * cam)

# ==================== ОСНОВНОЙ ЦИКЛ ====================

cap = cv2.VideoCapture(0)
print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Получаем CAM и предсказание
    cam, class_idx, prob = get_cam_and_prediction(frame)
    
    if cam is not None:
        # Масштабируем CAM до размера кадра
        cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # Накладываем на оригинал
        result = cv2.addWeighted(heatmap, 0.3, frame, 0.7, 0)
    else:
        result = frame
    
    # Добавляем текст с классом
    class_name = classes[class_idx] if class_idx < len(classes) else f"Class #{class_idx}"
    text = f"{class_name}: {prob:.2%}"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('CAM with Webcam', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
    print("Программа завершена")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрерывание пользователя")
        cv2.destroyAllWindows()