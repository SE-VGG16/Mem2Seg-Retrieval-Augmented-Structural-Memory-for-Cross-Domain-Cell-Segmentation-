import os
from PIL import Image
from torch.utils.data import Dataset


def find_mask_path(mask_dir, image_name):
    import os

    stem = os.path.splitext(image_name)[0].strip().lower().replace(" ", "")
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for fname in os.listdir(mask_dir):
        fstem, fext = os.path.splitext(fname)
        if fext.lower() not in exts:
            continue

        norm_fstem = fstem.strip().lower().replace(" ", "")
        if norm_fstem == stem:
            return os.path.join(mask_dir, fname)

    return None

    candidates = []

    # same stem, different extension
    for ext in exts:
        candidates.append(os.path.join(mask_dir, stem + ext))

    # common naming patterns
    for ext in exts:
        candidates.append(os.path.join(mask_dir, stem + "_mask" + ext))
        candidates.append(os.path.join(mask_dir, stem + "_label" + ext))
        candidates.append(os.path.join(mask_dir, stem + "_labels" + ext))
        candidates.append(os.path.join(mask_dir, stem + "_instance" + ext))
        candidates.append(os.path.join(mask_dir, stem + "_inst" + ext))

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(exts)
        ]
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), p


class SegmentationFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img, transform_mask):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.samples = []

        for name in sorted(os.listdir(image_dir)):
            if not name.lower().endswith(exts):
                continue

            mask_path = find_mask_path(mask_dir, name)
            if mask_path is not None:
                self.samples.append((name, mask_path))
            else:
                print(f"Warning: mask missing for {name}, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, mask_path = self.samples[idx]

        img_path = os.path.join(self.image_dir, name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_t = self.transform_img(img)
        mask_t = self.transform_mask(mask)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t, name