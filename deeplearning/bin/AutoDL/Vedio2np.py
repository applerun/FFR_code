import cv2 as cv
import os
import numpy as np

from PIL import Image
import PIL
from deeplearning.utils.VedioHandler.FrameTransform import *

dataroot = os.path.join("..", "..", "..", "..", "FFR_data")
video_src_src_path = os.path.join(dataroot, "CAG_raw")  # 数据集路径
label_names = os.listdir(video_src_src_path)
label2dir = {}
index = 0
size = (224, 224)
imshow = True


def save_vedio(frames, dst):
    frame_width, frame_height = frames[0].shape[0], frames[0].shape[1]
    out = cv.VideoWriter(dst)
    for frame in frames:
        out.write(frame)
    out.release()


def random_save(frames, dst):
    f = frames[len(frames) // 2]
    imsave(dst, f)


for label_name in label_names:
    if label_name.startswith('.'):
        continue
    label2dir[label_name] = index
    index += 1
    video_src_dir = os.path.join(video_src_src_path, label_name)
    video_save_dir = os.path.join(video_src_src_path + '_jpg', label_name)

    # if os.path.exists(video_save_dir):
    #     os.remove(video_save_dir)
    # os.makedirs(video_save_dir)

    videos = os.listdir(video_src_dir)
    # 过滤出avi文件
    videos = list(filter(lambda x: x.endswith('avi'), videos))
    lastcount = 0
    count = 0

    for each_video in videos:
        print("{}:{}/{}".format(label_name, count, len(videos)))
        count += 1
        if count <= lastcount:
            continue
        each_video_name, _ = each_video.split('.')

        video_save_each_video_dir = os.path.join(video_save_dir, each_video_name)
        if not os.path.exists(video_save_each_video_dir):
            os.makedirs(video_save_each_video_dir)

        each_video_save_full_path = os.path.join(video_save_dir, each_video_name)

        each_video_full_path = os.path.join(video_src_dir, each_video)

        cap = cv.VideoCapture(each_video_full_path)
        frames = []

        success = True
        while success:
            success, frame = cap.read()
            # print('read a new image:', success)
            if success:
                # img = Image.fromarray(image)
                # img = img.resize(size,Image.ANTIALIAS)
                frames.append(frame)
        if len(frames) == 0:
            continue
        frames = batch_transform_plot(frames, 400, plot = False)  # TODO:frames批处理

        for frame_count in range(len(frames)):
            frame = frames[frame_count]
            frame = frame.astype("float")
            frame = cv.resize(frame, size)
            dstname = os.path.join(each_video_save_full_path, "%d" % (frame_count * 1))
            if os.path.exists(dstname):
                os.remove(dstname)
            # if size is not None:
            #     frame = cv.resize(frame, size)
            # image = np.transpose(image,(2,0,1))
            # np.save(dstname, B * 0.114 + G * 0.387 + R * 0.299)

            np.save(dstname, frame)
            if imshow:
                imsave(dstname + ".png", frame)
            elif os.path.exists(dstname + ".png"):
                os.remove(dstname + ".png")
        random_save(frames, each_video_save_full_path + ".png")
        cap.release()
print(label2dir)
