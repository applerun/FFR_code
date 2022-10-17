# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import numpy as np
import matplotlib.pyplot as plt

dataroot = os.path.join("..", "FFR_data")


video_src_src_path = os.path.join(dataroot, "CAG_raw_jpg")  # 数据集路径
label_names = os.listdir(video_src_src_path)
label2dir = {}
index = 0
for label_name in label_names:
    if label_name.startswith('.'):
        continue
    label2dir[label_name] = index
    index += 1
    video_src_dir = os.path.join(video_src_src_path, label_name)

    video_dirs = os.listdir(video_src_dir)
    # 过滤出avi文件
    pshape = None
    for video_dir in video_dirs:
        video_dir_abs = os.path.join(video_src_dir, video_dir)
        for jpg in os.listdir(video_dir_abs):
            if not jpg.endswith(".npy"):
                continue
            jpg_abs = os.path.join(video_dir_abs, jpg)
            p = np.load(jpg_abs)
            print(jpg_abs, " loaded")
            # plt.imshow(p)
            # plt.show()
            # plt.close()
            if pshape is None:
                pshape = p.shape
            else:
                assert pshape == p.shape