import os
import shutil
import numpy as np
import pickle
import cv2
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def parse_img(img_dict: dict):
    print(img_dict.keys())
    # print(np.unique(np.array([lbl for lbl in img_dict[b"coarse_labels"]])))
    filenames = [img.decode("utf-8") for img in img_dict[b"filenames"]]
    imgs = img_dict[b"data"]

    return filenames, imgs

def save_img(filename, img, sub_dir="train"):
    path = os.path.join(root_dir, sub_dir, filename)
    img = np.array(img)
    shape = int(img.shape[-1] / 3)
    img = img.reshape(int(np.sqrt(shape)), -1, 3)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    pickle_root_dir = "D:/datasets/cifar100_python"
    train_data_dict = unpickle(os.path.join(pickle_root_dir, "train"))
    test_data_dict = unpickle(os.path.join(pickle_root_dir, "test"))

    root_dir = "D:/datasets/cifar/cifar100"
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    if os.path.exists(train_dir) or os.path.exists(test_dir):
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
    os.makedirs(train_dir, mode=777, exist_ok=True)
    os.makedirs(test_dir, mode=777, exist_ok=True)

    train_files, train_imgs = parse_img(train_data_dict)
    test_files, test_imgs = parse_img(test_data_dict)
    train_arr = np.array(list(zip(train_files, train_imgs)))
    test_arr = np.array(list(zip(test_files, test_imgs)))

    print(f":: Log :: [Data Shape] train-{train_arr.shape}, test-{test_arr.shape}")

    for file, img in train_arr:
        save_img(file, img)
    for file, img in test_arr:
        save_img(file, img, sub_dir="test")

