import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2

# ==============================
# PATHS
# ==============================
MODEL_PATH = "best_face_bbox_celeba_10k.pth"
IMAGE_PATH = r"C:/Users/Sabina/Desktop/dadam.jpg"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# MODEL (same as training)
# ==============================
class FaceBBoxRegressor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        backbone = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = backbone.features

        self.face_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        raw = self.face_head(feat)

        cx = raw[:, 0]
        cy = raw[:, 1]
        w = raw[:, 2] * 0.90
        h = raw[:, 3] * 0.60

        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)

        return torch.stack([x1, y1, x2, y2], dim=1)


# ==============================
# TRANSFORM (same as validation)
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==============================
# LOAD MODEL
# ==============================
model = FaceBBoxRegressor(pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ==============================
# LOAD IMAGE
# ==============================
img_pil = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = img_pil.size

input_tensor = transform(img_pil).unsqueeze(0).to(device)

# ==============================
# INFERENCE
# ==============================
with torch.no_grad():
    pred_box = model(input_tensor)[0].cpu().numpy()

# normalized → pixel
x1 = int(pred_box[0] * orig_w)
y1 = int(pred_box[1] * orig_h)
x2 = int(pred_box[2] * orig_w)
y2 = int(pred_box[3] * orig_h)

print(f"Predicted box: {x1}, {y1}, {x2}, {y2}")


# ==============================
# DRAW WITH OPENCV
# ==============================
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# draw bbox
cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

# label
cv2.putText(img_cv, "Face", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# show window
cv2.imshow("Face Detection Result", img_cv)

# wait until key press
cv2.waitKey(0)
cv2.destroyAllWindows()