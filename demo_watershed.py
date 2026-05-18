import cv2, numpy as np
from pathlib import Path

def run(img_path, out_dir="outputs_watershed"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1,3)).astype(np.float32)

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels_k, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels_k = labels_k.flatten().reshape((h,w))

    bg_cluster = int(np.argmax(centers[:,0]))
    mask_cells = np.uint8(labels_k != bg_cluster) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_cells = cv2.morphologyEx(mask_cells, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cells = cv2.morphologyEx(mask_cells, cv2.MORPH_CLOSE, kernel, iterations=2)

    dist = cv2.distanceTransform(mask_cells, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35*dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(mask_cells, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_ws = cv2.watershed(img.copy(), markers)
    boundary = (markers_ws == -1).astype(np.uint8)*255
    boundary = cv2.dilate(boundary, kernel, iterations=1)

    overlay = img.copy()
    overlay[boundary == 255] = (0,255,255)

    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(out/"cells_mask.png"), mask_cells)
    cv2.imwrite(str(out/"overlay_boundary.png"), overlay)
    print("Saved:", out.resolve())

if __name__ == "__main__":
    run("72128fd6-c4cd-468d-b941-6ca5965d1436.png")