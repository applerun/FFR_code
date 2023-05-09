import copy

import PIL.Image
import torch
import tqdm
from torch.utils.data import Dataset, Subset
import os, random, csv, glob
import numpy as np
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim = dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim = 0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        lens = [x[0].shape[self.dim] for x in batch]
        max_len = max(lens)
        newl = list(zip(lens, batch))
        newl = sorted(newl, key = lambda x: x[0], reverse = True)
        newl = zip(*newl)
        lens, batch = newl
        # pad according to max_len
        batch = [(pad_tensor(x, pad = max_len, dim = self.dim), y) for (x, y) in batch]
        # stack all
        xs = torch.stack(tuple(x[0] for x in batch), dim = 0)
        ys = torch.LongTensor(tuple(x[1] for x in batch))
        return xs, ys, lens

    def __call__(self, batch):
        xs, ys, lengths = self.pad(batch)
        xs = pack_padded_sequence(xs, lengths, batch_first = True)
        return xs, ys


class FFRDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 mode = "train",
                 t_v_t = None,
                 sfpath = "videos.csv",
                 shuffle = True,
                 transform = None,
                 backEnd = ".npy",
                 unsupervised: bool = False,
                 noising = None,
                 newfile = False,
                 k_split: int = None,
                 k: int = 0,
                 ratio: dict = None,
                 timesteps = None,
                 timesteps_random = False,
                 max_timestep = -1,
                 ):
        """

        :param dataroot: 数据根目录
        :param mode: "train":训练集 "val":验证集 "test":测试集
        :param t_v_t:[float,float,float] 分割所有数据train-validation-test的比例
        :param sfpath: 在啊数据根目录创建记录所有数据的文件
        :param shuffle: 是否将读取的数据打乱
        :param transform: 数据预处理/增强
        :param backEnd: 数据文件后缀
        :param unsupervised: 是否为无监督学习
        :param noising: 加噪声
        :param newfile: 不沿用已有的sf，重新创建数据集
        :param k_split: k-折交叉验证
        :param k: 第k次k-折交叉验证
        :param ratio: 降/过采样
        :param timesteps: 视频长度
        :param timesteps_random: 随机截取视频位置
        """
        super(FFRDataset, self).__init__()
        self.timesteps = timesteps
        self.timesteps_random = timesteps_random
        if t_v_t is None and k_split is None:  # 分割train-validation-test
            t_v_t = [0.7, 0.2, 0.1]
            # if type(t_v_t) is list:
            # 	t_v_t = numpy.array(t_v_t)
        if transform is None:
            transform = transforms.ToTensor()
        self.k_split = k_split
        if k_split is not None:  # k_split 模式
            # t_v_t = [x*k_split for x in t_v_t]
            assert 0 <= k < k_split, "k must be in range [{},{}]".format(0, k_split - 1)
        self.k = k
        # assert t_v_t[0] + t_v_t[1] <= 1
        self.tvt = t_v_t
        self.new = newfile
        self.dataroot = dataroot
        self.name2label = {}  # 为每个分类创建一个label
        self.sfpath = sfpath
        self.shuff = shuffle
        self.mode = mode
        self.dataEnd = backEnd
        self.transform = transform
        self.unsupervised = unsupervised
        self.noising = noising
        self.train_split = self.tvt[0]
        self.validation_split = self.tvt[1]
        self.test_split = self.tvt[2]
        self.xs = None
        self.imgDirs = []
        self.labels = []
        self.ratio = ratio
        self.max_timestep = max_timestep
        for name in sorted(os.listdir(dataroot)):
            if not os.path.isdir(os.path.join(dataroot, name)):
                continue
            if not len(os.listdir(os.path.join(dataroot, name))):
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.numclasses = len(self.name2label.keys())
        self.LoadCsv(sfpath)  # 加载所有的数据文件
        # 数据分割
        self.split_data()
        return

    def LoadCsv(self, sfpath):
        header = ["label", "filepath"]

        if not os.path.exists(os.path.join(self.dataroot, sfpath)) or self.new:
            imgDirs = []
            for name in self.name2label.keys():
                dirs = glob.glob(os.path.join(self.dataroot, name, "*"))

                dirs = list(filter(lambda x: os.path.isdir(x) and len(os.listdir(x)) > 1, dirs))
                if self.ratio is not None:
                    if not name in self.ratio.keys():
                        ratio = 1.0
                    else:
                        ratio = self.ratio[name]

                    if ratio < 1.0:
                        dirs = random.sample(dirs, int(ratio * len(dirs)))
                    elif ratio > 1.0:
                        for dir in dirs:
                            if random.random() * ratio > 1:
                                dirs.append(dir)
                        pass
                imgDirs += dirs

            if self.shuff:  # 打乱顺序
                random.shuffle(imgDirs)

            with open(os.path.join(self.dataroot, sfpath), mode = "w", newline = "") as f:  # 记录所有数据
                writer = csv.writer(f)
                writer.writerow(header)

                for imgDir in imgDirs:  # imgDir:data root/label name/**.csv
                    name = imgDir.split(os.sep)[-2]  # label name
                    label = self.name2label[name]  # label idx

                    writer.writerow([label, imgDir])

        self.imgDirs = []
        self.labels = []

        with open(os.path.join(self.dataroot, sfpath)) as f:
            reader = csv.reader(f)
            for row in reader:
                if row == header:
                    continue
                try:
                    label = int(row[0])
                    imgDir = row[1]
                    self.labels.append(torch.tensor(label))
                    self.imgDirs.append(imgDir)
                except:  # 数据格式有误
                    print("wrong csv,remaking...")
                    f.close()
                    os.remove(os.path.join(self.dataroot, sfpath))
                    self.LoadCsv(sfpath)
                    break
        assert len(self.imgDirs) == len(self.labels)
        return self.imgDirs, self.labels  # [[float],[],...[]],[int]

    def split_data(self):
        train_split_int = int(self.train_split * len(self.imgDirs))
        val_split_int = int((self.train_split + self.validation_split) * len(self.imgDirs))

        if self.mode == "test":  # 分割测试集
            self.imgDirs = self.imgDirs[val_split_int:]
            self.labels = self.labels[val_split_int:]
            return

        if self.k_split is None:
            if self.mode == "train" and self.k_split is None:  # 分割训练集
                self.imgDirs = self.imgDirs[:train_split_int]
                self.labels = self.labels[:train_split_int]
            elif self.mode == "val" and self.k_split is None:  # 分割验证集
                self.imgDirs = self.imgDirs[train_split_int:val_split_int]
                self.labels = self.labels[train_split_int:val_split_int]
            else:
                raise Exception("Invalid mode!", self.mode)
        else:
            self.imgDirs = self.imgDirs[:val_split_int]
            self.labels = self.labels[:val_split_int]
            val_start = int(round(self.k * len(self.imgDirs) / self.k_split))
            val_end = int(round((self.k + 1) * len(self.imgDirs) / self.k_split))

            # l = list(set(l1) & set(l2))
            if self.mode == "train":
                self.imgDirs = self.imgDirs[:val_start] + self.imgDirs[val_end:]
                self.labels = self.labels[:val_start] + self.labels[val_end:]
            elif self.mode == "val":
                self.imgDirs = self.imgDirs[val_start:val_end]
                self.labels = self.labels[val_start:val_end]
        return self.imgDirs, self.labels

    def shufflecsv(self,
                   filename = None):
        if filename is None:
            filename = self.sfpath
        path = os.path.join(self.dataroot, filename)
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append(row)
            random.shuffle(rows)
        with open(path, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return

    def shuffle(self):
        z = list(zip(self.imgDirs, self.labels))
        random.shuffle(z)
        self.imgDirs, self.labels = zip(*z)
        return

    def num_classes(self):
        return self.numclasses

    def name2label(self):
        return self.name2label

    def label2name(self):
        keys = list(self.name2label.values())
        values = list(self.name2label.keys())
        return dict(zip(keys, values))

    def get_data_sorted_by_label(self):
        """
        返回一个字典，每个label对应的数字指向一个tensor[label=label的光谱的数量,c=1,lenth]
        """
        data_each_label = {}
        for i in range(len(self.name2label)):
            if self.timesteps is not None:
                data_each_label[i] = None
            else:
                data_each_label[i] = []

        for i in range(self.__len__()):
            video, label = self[i]
            print("\r{}/{}".format(i + 1, len(self)), end = "")
            if self.timesteps is not None:
                video = torch.unsqueeze(video, dim = 0)
                if data_each_label[label.item()] is None:
                    data_each_label[label.item()] = video
                else:
                    data_each_label[label.item()] = torch.cat(
                        (data_each_label[label.item()],
                         video),
                        dim = 0
                    )
            else:
                data_each_label[label.item()].append(video)
        print("load_done")
        if self.timesteps is None:
            print("{}/{}".format(0, len(self.name2label)), end = "")
            for i in range(len(self.name2label)):
                data_each_label[i] = [(data_each_label[i][x], torch.tensor(i)) for x in data_each_label[i]]
                data_each_label[i], _, lens = PadCollate().pad(data_each_label[i])
                data_each_label[i] = pack_padded_sequence(data_each_label[i], lens)
                print("\r{}/{}".format(i + 1, len(self.name2label)), end = "")
            print("")
        for k in data_each_label.keys():
            if data_each_label[k] is None:
                continue
            assert len(data_each_label[k].shape) == 5  # [b , t , c = 3, h ,w ]
        return data_each_label

    def savedata(self,
                 dir,
                 mode = "file_wise"):
        label2data = self.get_data_sorted_by_label()
        if not os.path.isdir(dir):
            os.makedirs(dir)
        if mode == "file_wise":
            for name in self.name2label.keys():
                path = os.path.join(dir, name + ".csv")
                data = label2data[self.name2label[name]]
                data = data.numpy()
                data = np.squeeze(data)
                np.savetxt(path, data, delimiter = ",")

    def __len__(self):
        return len(self.imgDirs)

    def __getitem__(self, item):
        imgDir = self.imgDirs[item]
        path2imgs = os.listdir(imgDir)
        path2imgs = list(filter(lambda x: x.endswith(self.dataEnd), path2imgs))

        if self.timesteps is "max":
            l = [len(list(filter(lambda x: x.endswith(self.dataEnd), os.listdir(dir2imgs)))) for dir2imgs in
                 self.imgDirs]
            timesteps = max(l)
            if timesteps > 1:
                self.timesteps = timesteps

        if self.timesteps_random and self.timesteps is not None:
            start_range = list(range(len(path2imgs) - self.timesteps))
            start = random.sample(start_range, 1)[0]
            path2imgs = path2imgs[start:start + self.timesteps]
        else:
            path2imgs = path2imgs[:self.timesteps]

        label = self.labels[item]

        frames = []
        for p2i in path2imgs:  # 读取img
            frame = np.load(os.path.join(imgDir, p2i))
            frame = frame.astype("float32")
            # image = np.transpose(image,(1,2,0))
            frames.append(frame)
            if len(frames) == self.max_timestep:
                break

        seed = np.random.randint(1e9)
        frames_tr = []
        for frame in frames:  # transform
            if self.transform is not None:
                random.seed(seed)
                np.random.seed(seed)
                frame = self.transform(frame)
            frames_tr.append(frame)

        if self.timesteps is not None and 0 < len(frames_tr) < self.timesteps:  # 长度补齐
            for i in range(self.timesteps - len(frames_tr)):
                frames_tr.append(copy.deepcopy(frames_tr[-1]))

        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)

        return frames_tr, label  # [t,l,w,c] or [t,l,w],[label]


if __name__ == '__main__':
    dataroot = "../../../FFR_data/CAG_raw_jpg"
    train_transformer = transforms.Compose([
        # transforms.ToPILImage(mode = "RGB"),
        # transforms.RandomHorizontalFlip(p = 0.5),
        # transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
        transforms.ToTensor(),
    ])

    from deeplearning.utils.DataLoader import MultiEpochsDataLoader as DataLoader

    dataset = FFRDataset(dataroot, transform = train_transformer)
    P = PadCollate()
    loader = tqdm.tqdm(DataLoader(dataset, shuffle = True, collate_fn = P, batch_size = 8, num_workers = 8))
    for step, (spectrum, label) in enumerate(loader):
        pass
