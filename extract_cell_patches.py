import os
import cv2
import numpy as np

img_dir = r"C:\Users\Sabina\Desktop\SABINAS_APPLICATION_DOCS\BK_21_Sabina\Cell_Biology\Code_files\tools\data\train\images"
mask_dir = r"C:\Users\Sabina\Desktop\SABINAS_APPLICATION_DOCS\BK_21_Sabina\Cell_Biology\Code_files\tools\data\train\masks"
out_dir = r"C:\Users\Sabina\Desktop\SABINAS_APPLICATION_DOCS\BK_21_Sabina\Cell_Biology\Code_files\tools\data\prototype_patches"

os.makedirs(out_dir, exist_ok=True)

count = 0

for name in os.listdir(img_dir):

    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name.replace(".jpg", ".png"))

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    # find connected components (cells)
    num_labels, labels = cv2.connectedComponents(mask)

    for i in range(1, num_labels):

        component = (labels == i).astype(np.uint8)

        ys, xs = np.where(component)

        if len(xs) < 20:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        patch = img[y1:y2, x1:x2]

        patch = cv2.resize(patch, (128,128))

        cv2.imwrite(os.path.join(out_dir, f"patch_{count}.png"), patch)

        count += 1

print("Total patches:", count)