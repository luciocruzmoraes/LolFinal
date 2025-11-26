import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "datasets/raw/skins"
TRAIN_DIR = "datasets/processed/train"
VAL_DIR = "datasets/processed/val"
TEST_DIR = "datasets/processed/test"

def split_dataset(test_size=0.2, val_size=0.2):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for cls in os.listdir(RAW_DIR):
        src = os.path.join(RAW_DIR, cls)

        if not os.path.isdir(src):
            continue
        images = [
            os.path.join(src, img)
            for img in os.listdir(src)
            if os.path.isfile(os.path.join(src, img))
        ]
        if len(images) < 3:
            print(f"[!] Classe '{cls}' só tem {len(images)} imagem(ns). Indo tudo para TRAIN.")
            cls_train_path = os.path.join(TRAIN_DIR, cls)
            os.makedirs(cls_train_path, exist_ok=True)
            for img in images:
                shutil.copy(img, cls_train_path)
            continue
        train_val, test_split = train_test_split(
            images, test_size=test_size, random_state=42
        )
        train_split, val_split = train_test_split(
            train_val,
            test_size=val_size/(1-test_size),
            random_state=42
        )
        cls_train = os.path.join(TRAIN_DIR, cls)
        cls_val   = os.path.join(VAL_DIR, cls)
        cls_test  = os.path.join(TEST_DIR, cls)

        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_val, exist_ok=True)
        os.makedirs(cls_test, exist_ok=True)

        for img in train_split:
            shutil.copy(img, cls_train)
        for img in val_split:
            shutil.copy(img, cls_val)
        for img in test_split:
            shutil.copy(img, cls_test)

    print("finalizado")
