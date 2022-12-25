import numpy as np
import cv2 as cv


def remove_black(frame: np.ndarray):
    if len(frame.shape) == 3:
        assert frame.shape[-1] in [3, 1]
    else:
        assert len(frame.shape) == 2

    left = 0
    sig = np.sum(frame[:, left, :] == 0) / frame.shape[0] / 3
    while sig > 0.5:
        left += 1
        sig = np.sum(frame[:, left, :] == 0) / frame.shape[0] / 3
    left += 5
    right = frame.shape[1]
    while np.sum(frame[:, right - 1, :] == 0) / frame.shape[0] / 3 > 0.5:
        right -= 1
    right -= 30
    frame = frame[:, left:right, :]
    return frame


def padding(frame, direction = (0, 1, 0, 1), fill = 0):
    """

    :param frame:
    :param direction:上下左右各延长多少
    :param fill: padding的内容
    :return:
    """
    assert len(direction) == 4
    shape = frame.shape
    res = np.ones((shape[0] + direction[0] + direction[1],
                   shape[1] + direction[2] + direction[3]))
    res *= fill
    res[direction[0]:direction[0] + shape[0], direction[2]:direction[2] + shape[1]] = frame
    return res


def conv_2d_point(frame_, x, y, core, padding = 0):
    if len(frame_.shape) == 3:
        assert frame_.shape[-1] in [3, 1]
    else:
        assert len(frame_.shape) == 2
    threeD = False
    if len(core.shape) == 3:
        if core.shape[2] == 3:
            threeD = True
        else:
            assert core.shape[2] == 1
            core = np.squeeze(core)
    else:
        assert len(frame_.shape) == 2
    lenx, leny = core.shape[0], core.shape[1]
    res = np.zeros((3))
    src = frame_[x:x + lenx, y:y + leny, :]
    if threeD:
        for i in range(3):
            res[i] = src[:, :, i] * core[i]
    else:
        for i in range(3):
            res[i] = src[:, :, i] * core
    return res


def tranform_series(l):
    def func(frame):
        for f in l:
            frame = f(frame)
        return frame

    return func


def rec(shape, mid, size):
    assert shape[0] >= size[0] and shape[1] >= size[1], "size of rec({}) must be less than the shape({})".format(size,
                                                                                                                 shape)
    res = (0, 0, *shape)

    left = max(mid[0] - int(size[0] / 2), 0)
    right = left + size[0]
    right = min(right, shape[0])
    if right == shape[0]:
        left = right - size[0]

    up = max(mid[1] - int(size[1] / 2), 0)
    down = up + size[1]
    down = min(down, shape[0])
    if down == shape[1]:
        up = down - size[1]

    return (left, up), (right, down)


def __remove_outlier(points):
    median = np.median(points, axis = 0)
    deviations = abs(points - median)
    mad = np.median(deviations, axis = 0)

    remove_idx = np.where(deviations > mad * 4)
    return np.delete(points, remove_idx, axis = 0)


def batch_transform_plot(images, minsize = 400, plot = True):
    mids = []
    ms = []
    new_images = []
    x_, y_, w_, h_ = 0, 0, 0, 0
    for image in images:
        image = remove_black(image)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray, 16, 80, apertureSize = 3, L2gradient = True)  # apertureSize参数默认其实就是3
        lines = cv.HoughLines(edges, 1, np.pi / 180, 80)
        xs = []
        ys = []

        for line in lines:
            rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。

            a = np.cos(theta)  # theta是弧度
            b = np.sin(theta)
            x0 = a * rho  # 代表x = r * cos（theta）
            y0 = b * rho  # 代表y = r * sin（theta）
            if theta == 0:
                xs.append(x0)
            elif np.abs(theta - np.pi / 2) < 1e-6:
                if y0 > image.shape[1] / 2:
                    ys.append(y0)
                # 画出检测到的横线
                if plot:
                    x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
                    y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
                    x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
                    y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
                    cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 点的坐标必须是元组，不能是列表。

            else:
                continue
        if len(ys) > 0:  # 有白线
            y_ = int(min(ys)) - 2
        else:  # 无白线
            y_ = edges.shape[1]
        # 画出检测到的绿线
        if plot:
            for x in xs:
                x = int(x)
                cv.line(image, (x, y_), (x, y_ + 1000), (0, 255, 255), 2)  # 点的坐标必须是元组，不能是列表。

        detection_image = edges[0:y_]
        moment = cv.moments(detection_image)

        detection_image_rgb = cv.cvtColor(detection_image, cv.COLOR_GRAY2RGB)  # 将轮廓化为RGB格式
        image = image[0:y_]
        if plot:
            image = image * (detection_image_rgb == 0) + detection_image_rgb * (225, 255, 255)  # 在原图中plot轮廓

        new_images.append(image)
        m = (detection_image != 0).sum()
        if m == 0:
            continue
        mid = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))  # 求质心

        mids.append(mid)
        if plot:
            cv.circle(image, mid, 15, (0, 0, 255))  # 在原图中plot质心
        ms.append(m)
        temp = detection_image != 0
        temp = temp.astype("float")
        points = cv.findNonZero(temp)
        points = np.squeeze(points, axis = 1)
        points = __remove_outlier(points)
        if len(points) < 10:
            continue
        x, y, w, h = cv.boundingRect(points)
        if w * h > w_ * h_:
            x_, y_, w_, h_ = x, y, w, h
        if plot:
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))  # 画出边缘伦奎

    images = new_images
    xy = [0, 0]
    for i in range(len(mids)):
        xy[0] += mids[i][0] * ms[i]
        xy[1] += mids[i][1] * ms[i]
    _sum = sum(ms)
    xy[0] /= _sum
    xy[1] /= _sum
    xy[0], xy[1] = int(xy[0]), int(xy[1])
    s = min(max(int((w_ * h_) ** 0.5 / 2 * 1.2), minsize), images[0].shape[0], images[0].shape[1])

    p1, p2 = rec(images[0].shape, xy, (s, s))
    p1 = (0, 0)
    p2 = (s, s)
    # 框选范围
    if plot:
        for image in images:
            cv.rectangle(image, p1, p2, (0, 0, 225))  # 在原图中plot根据质心画出的框选范围
    else:
        for i in range(len(images)):
            image = images[i]
            images[i] = image[p1[0]:p2[0], p1[1]:p2[1], :]
    return images


def transform_default(image, size = (500, 500)):
    image = remove_black(image)

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 80, 100, apertureSize = 3)  # apertureSize参数默认其实就是3

    lines = cv.HoughLines(edges, 1, np.pi / 180, 80)
    xs = []
    ys = []
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。

        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        if theta == 0:
            xs.append(x0)
        elif np.abs(theta - np.pi / 2) < 1e-6:
            ys.append(y0)
            # 画出检测到的横线
            # x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
            # y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
            # x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
            # y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
            # # cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 点的坐标必须是元组，不能是列表。

        else:
            continue
    if len(ys) > 0:  # 有白线
        y_ = int(min(ys)) - 2
    else:  # 无白线
        y_ = edges.shape[1]
    # 画出检测到的绿线
    # for x in xs:
    #     x = int(x)
    #     cv.line(image, (x, y_), (x, y_ + 1000), (0, 255, 255), 2)  # 点的坐标必须是元组，不能是列表。

    detection_image = edges[0:y_]
    moment = cv.moments(detection_image)
    mid = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))  # 求质心
    detection_image = cv.cvtColor(detection_image, cv.COLOR_GRAY2RGB)  # 将轮廓化为RGB格式
    res_image = image[0:y_]
    res_image = res_image * (detection_image == 0) + detection_image * (225, 255, 255)  # 在原图中plot轮廓
    cv.circle(res_image, mid, 15, (0, 0, 255))  # 在原图中plot质心
    p1, p2 = rec(res_image.shape, mid, size)

    cv.rectangle(res_image, p1, p2, (0, 0, 225))
    return res_image


def imread(path):
    image_array = cv.imdecode(np.fromfile(path, dtype = np.uint8), -1)
    return image_array


# 存文件
def imsave(path, image):
    cv.imencode('.jpg', image)[1].tofile(path)  # .jpg, .png


if __name__ == '__main__':
    import os

    imshow = True
    size = (224, 224)
    dataroot = os.path.join("..", "..", "..", "..", "FFR_data")
    video_sample_path = os.path.join(dataroot, "CAG_raw", "001-050", '18苏殿明.avi')
    dst_sample_path = os.path.join(dataroot, "CAG_raw_png", "001-050", '18苏殿明')
    if not os.path.isdir(dst_sample_path):
        os.makedirs(dst_sample_path)
    cap = cv.VideoCapture(video_sample_path)
    frames = []

    success = True
    while success:
        success, frame = cap.read()
        # print('read a new image:', success)
        if success:
            # img = Image.fromarray(image)
            # img = img.resize(size,Image.ANTIALIAS)
            frames.append(frame)
    frames = batch_transform_plot(frames)
    for frame_count in range(len(frames)):
        frame = frames[frame_count]
        dstname = os.path.join(dst_sample_path, "%d" % (frame_count * 1))
        if os.path.exists(dstname):
            os.remove(dstname)
        np.save(dstname, frame)
        if imshow:
            imsave(dstname + ".png", frame)
        elif os.path.exists(dstname + ".png"):
            os.remove(dstname + ".png")
    cap.release()
