from torch.utils.data import Dataset, DataLoader, Subset
import os

class FFRDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 mode="train",
                 t_v_t=None,
                 sfpath="videos.csv",
                 shuffle=True,
                 transform=None,
                 backEnd=".npy",
                 unsupervised: bool = False,
                 noising=None,
                 newfile=False,
                 k_split: int = None,
                 k: int = 0,
                 ratio: dict = None,
                 timesteps = None,
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
        """
        super(FFRDataset,self).__init__()
        self.timesteps = timesteps
        if t_v_t is None and k_split is None:  # 分割train-validation-test
            t_v_t = [0.7, 0.2, 0.1]
            # if type(t_v_t) is list:
            # 	t_v_t = numpy.array(t_v_t)

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
        self.RamanFiles = []
        self.labels = []
        self.Datas = []
        self.ratio = ratio

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
        self.load_raman_data()  # 数据读取
        return

