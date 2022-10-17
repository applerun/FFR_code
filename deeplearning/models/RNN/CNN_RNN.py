import torch
import torchvision.models
from torch import nn
from deeplearning.models.BasicModule import *


class BaseModelRNN(BasicModule):
    default_init = dict(
        num_classes = 2, dr_rate = 0.2, num_hiddens = 100,
        basenet = "torchvision.models.resnet18(pretrained = False)",
        rnn = nn.LSTM, num_features = 100,
        rnn_num_layers = 1,
    )

    def __init__(self, num_classes, num_hiddens, num_features=512, rnn_num_layers=1, basenet = None, rnn = None,
                 dr_rate = 0.2):
        """
        :param num_classes:  最终输出size（分类数）
        :param dr_rate: dropout rate
        :param num_hiddens: 隐层数
        :param basenet: CNN net：输出数为num_features
        :param rnn: Class of RNN net :输入为num_features 输出为num_hiddens
        :param num_features: CNN输入RNN的size
        :param rnn_num_layers: rnn层数

        """

        super(BaseModelRNN, self).__init__()
        self.rnn_num_layers = rnn_num_layers
        self.num_features = num_features
        self.num_hiddens = num_hiddens

        if rnn is None:
            self.create_rnn()
        else:
            self.rnn = rnn(input_size = num_features, hidden_size = num_hiddens, num_layers = rnn_num_layers)

        if basenet is None:
            self.create_baseCNNnet()
        else:
            self.basenet = basenet
        self.dropOut = nn.Dropout(dr_rate)
        self.fc1 = nn.Linear(num_hiddens, num_classes)
        self.flatten = Flat()
        self.model_name = str(type(self)) + "_" + str(type(self.basenet)) + "_" + str(type(self.rnn))

    def create_baseCNNnet(self):
        self.basenet = torchvision.models.resnet18(pretrained = False)
        self.basenet.fc = Identity()
        return

    def create_rnn(self):
        self.rnn = nn.LSTM(input_size = self.num_features, hidden_size = self.num_hiddens,
                           num_layers = self.rnn_num_layers)
        return

    def forward(self, x):
        if len(x.shape) == 4:
            bs, ts, h, w = x.shape
        elif len(x.shape) == 5:
            bs, ts, c, h, w = x.shape
        else:
            print("unknown x shape")
            raise ValueError
        tt = 0
        y = self.basenet((x[:, tt]))  # [b ]
        # out, (hn, cn) = self.rnn(y.unsqueeze(1))  # []
        out, hncn = self.rnn(y.unsqueeze(1))  # []
        for tt in range(1, ts):
            y = self.basenet((x[:, tt]))
            # out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            out, hncn = self.rnn(y.unsqueeze(1), hncn)
        # out = self.dropOut(out[:,-1])
        out = self.flatten(out)
        out = self.dropOut(out)
        out = self.fc1(out)

        return out


class RNNModel(BasicModule):
    def __init__(self, num_hiddens, input_size, ):
        super(RNNModel, self).__init__()
        # # 定义RNN层
        # 输入的形状为（num_steps, batch_size, input_size）  # input_size 就是 vocab_size
        # 输出的形状为（num_steps, batch_size, num_hiddens）
        self.rnn = nn.GRU(input_size, num_hiddens)
        self.input_size = self.rnn.input_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.input_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.input_size)

    def forward(self, inputs, state):
        # inputs的形状为（num_steps, batch_size, input_size）
        # Y是所有时间步的隐藏状态，state是最后一个时间步的隐藏状态
        # Y的形状为（num_steps, batch_size, hidden_size），state为（1，batch_size, hidden_size）
        Y, state = self.rnn(inputs, state)
        # 全连接层首先将Y的形状改为(num_steps*batch_size, hidden_size)
        # 它的输出形状是(num_steps*batch_size,input_size)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size = 1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                               device = device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device = device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device = device))


class Resnet18LSTM(BaseModelRNN):
    def __init__(self, num_classes, num_hiddens=100, num_features = 512, rnn_num_layers=1, dr_rate = 0.2):
        super(Resnet18LSTM, self).__init__(num_classes, num_hiddens, num_features, rnn_num_layers,
                                           dr_rate = dr_rate)


class Resnet18GRU(BaseModelRNN):
    def __init__(self, num_classes, num_hiddens=100, num_features = 512, rnn_num_layers=1, dr_rate = 0.2):
        super(Resnet18GRU, self).__init__(num_classes, num_hiddens, num_features, rnn_num_layers, dr_rate = dr_rate)

    def create_rnn(self):
        self.rnn = nn.GRU(input_size = self.num_features, hidden_size = self.num_hiddens,
                          num_layers = self.rnn_num_layers)


if __name__ == '__main__':
    input = torch.zeros((4, 60, 3, 224, 224))
    net0 = Resnet18LSTM(2)
    net1 = Resnet18GRU(2)
    out0 = net0(input)
    out1 = net1(input)
    print(out0.shape)
    print(out1.shape)
