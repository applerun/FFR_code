import os
import numpy as np
import cv2
from PIL import Image
import PIL
dataroot = os.path.join("..", "..", "..","..", "FFR_data")

video_src_src_path = os.path.join(dataroot, "CAG_raw")  # 数据集路径
label_names = os.listdir(video_src_src_path)
label2dir = {}
index = 0
size = (224,224)
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

    count = 0
    for each_video in videos:
        print("{}:{}/{}".format(label_name,count,len(videos)))
        count+=1
        each_video_name, _ = each_video.split('.')

        video_save_each_video_dir = os.path.join(video_save_dir, each_video_name)
        if not os.path.exists(video_save_each_video_dir):
            os.mkdir(video_save_each_video_dir)

        each_video_save_full_path = os.path.join(video_save_dir, each_video_name)

        each_video_full_path = os.path.join(video_src_dir, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            # print('read a new frame:', success)
            dstname = os.path.join(each_video_save_full_path, "%d" % (frame_count * 1))
            if os.path.exists(dstname):
                os.remove(dstname)
            if success:
                # img = Image.fromarray(frame)
                # img = img.resize(size,Image.ANTIALIAS)
                frame = cv2.resize(frame,size)
                # frame = np.transpose(frame,(2,0,1))

                # np.save(dstname, B * 0.114 + G * 0.387 + R * 0.299)
                np.save(dstname, frame)

            frame_count += 1
        cap.release()
np.save('label_dir.npy', label2dir)
print(label2dir)
