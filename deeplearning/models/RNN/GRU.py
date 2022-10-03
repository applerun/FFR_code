import torch
from torch import nn
from deeplearning.models.BasicModule import BasicModule
class RNNModel(BasicModule):
    def __init__(self, num_hiddens, input_size,):
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

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))