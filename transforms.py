import random
import numpy as np
from PIL import Image
import torchvision.transforms as T

def build_basic_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

def build_pretrain_augment(image_size=224):
    # simple augmentation for contrastive learning
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.1)], p=0.7),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.ToTensor(),
    ])