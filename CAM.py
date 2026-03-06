
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import json
import urllib.request
from PIL import Image


# Метки классов
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    classes = [item[1] for item in json.loads(response.read().decode()).values()]

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
    nc, h, w = feature_conv.shape[1:]
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return np.uint8(255 * cam)


cap = cv2.VideoCapture(0)
print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cam, class_idx, prob = get_cam_and_prediction(frame)
    
    if cam is not None:
        cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        result = cv2.addWeighted(heatmap, 0.3, frame, 0.7, 0)
    else:
        result = frame
    
    class_name = classes[class_idx] if class_idx < len(classes) else f"Class #{class_idx}"
    text = f"{class_name}: {prob:.2%}"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('CAM', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрерывание пользователя")
        cv2.destroyAllWindows()