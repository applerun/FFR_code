import numpy
import torch
import sys, os
import time
import numpy as np
import torch.nn.functional as F
import visdom
from sklearn.metrics import roc_curve, auc, confusion_matrix

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )


# __all__ = ["evaluate","startVisdomServer","evaluate_criteon","data2mean_std","spectrum_vis"]
def evaluate_all(model,
                 loader,
                 criteon,
                 device: torch.device,
                 num_classes = None,
                 loss_plus = None):
    """

    验证当前模型在验证集或者测试集的准确率\ROC\ TODO：分别验证不同label的准确度并可视化

    """
    model.eval()
    correct = 0
    total = len(loader.dataset)
    loss_list = []
    if num_classes is None:
        num_clases = loader.dataset.num_classes()
    y_true_all = {}
    y_score_all = {}
    conf_m = np.zeros((num_clases, num_clases))
    for i in range(num_clases):
        y_true_all[i] = numpy.array([])
        y_score_all[i] = numpy.array([])
    if total == 0:
        return 0
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            output = model(x)  # [b,n_c]
            pred = output.argmax(dim = 1)  # []

            c_t = torch.eq(pred, y).sum().float().item()
            correct += c_t
            loss = criteon(output, y) if loss_plus is None else criteon(output[0], y) + loss_plus(
                *output[1:])
            loss_list.append(loss.item())
            score = F.softmax(output, dim = 1)  # TODO:也许score就设置为output？
            for i in range(num_clases):
                y_true = torch.eq(y, i).int().cpu().numpy()  # [b]
                y_score = score[:, i].float().cpu().numpy()  # [b]
                y_true_all[i] = np.append(y_true_all[i], y_true)
                y_score_all[i] = np.append(y_score_all[i], y_score)
            cm = confusion_matrix(y.cpu().numpy(), pred.cpu().numpy(), labels = list(range(num_clases)))
            conf_m += cm

    label2roc = {}
    label2auc = {}
    for i in range(num_clases):
        frp, tpr, thresholds = roc_curve(y_true_all[i], y_score_all[i])
        label2roc[i] = (frp, tpr, thresholds)
        label2auc[i] = auc(frp, tpr)

    acc = correct / total
    loss = np.mean(loss_list)

    res = dict(acc = acc, loss = loss, label2roc = label2roc, label2auc = label2auc, confusion_matrix = conf_m)
    return res


def evaluate(model,
             loader,
             device: torch.device):
    """

    验证当前模型在验证集或者测试集的准确率 TODO：分别验证不同label的准确度并可视化

    """
    model.eval()
    correct = 0
    total = len(loader.dataset)
    if total == 0:
        return 0
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim = 1)

        c_t = torch.eq(pred, y).sum().float().item()
        correct += c_t
    acc = correct / total
    return acc


# def evaluate_labelwise(model,
#                        dataset: FFRDataset,
#                        device: torch.device,
#                        viz: visdom.Visdom = None):
#     model.eval()
#     label2data = dataset.get_data_sorted_by_label()
#     label2name = dataset.label2name()
#     label2acc = {}
#     for k in label2data.keys():
#         data = label2data[k]
#         name = label2name[k]
#         labels = torch.ones(data.shape[0]) * k
#
#         data, labels = data.to(device), labels.to(device)
#         with torch.no_grad():
#             logits = model(data)
#             pred = logits.argmax(dim = 1)
#         c_t = torch.eq(pred, labels).sum().float().item()
#         label2acc[name] = c_t / pred.shape[0]
#     return label2acc
#
#
# def evaluate_samplewise(model,
#                         dataset: Raman_dirwise or dict,
#                         device: torch.device):
#     # TODO:
#     t0 = time.time()
#     model.eval()
#     sample2acc = {}
#     sample2label = dataset.sample2label()
#     sample2data = dataset.get_data_sorted_by_sample()
#     t1 = time.time()
#     if t1 - t0 > 1:
#         print("get_data_time:", t1 - t0)
#     for k in sample2data.keys():
#         t0 = time.time()
#         data = sample2data[k]
#         labels = torch.ones(data.shape[0]) * sample2label[k]
#         data, labels = data.to(device), labels.to(device)
#         with torch.no_grad():
#             logits = model(data)
#             pred = logits.argmax(dim = 1)
#         c_t = torch.eq(pred, labels).sum().float().item()
#         sample2acc[k] = c_t / pred.shape[0]
#         t1 = time.time()
#         if t1 - t0 > 1:
#             print("cal_acc_time_{}:".format(k), t1 - t0)
#     return sample2acc


def evaluate_loss(model,
                  loader,
                  criteon,
                  device,
                  loss_plus: callable = None):
    """
    验证当前模型在验证集或者测试集的准确率
    """
    loss_list = []
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            output = model(x)
            loss = criteon(output, y) if loss_plus is None else criteon(output[0], y) + loss_plus(
                *output[1:])
            loss_list.append(loss.item())
    return np.mean(loss_list)


# def comp_class_vec(output: torch.Tensor,
#                    num_classes,
#                    index = None,
#                    device = None
#                    ):
#     """
#
#     :param output:[b,n_c]
#     :param index: [b] or int or []
#     :return: class_vec
#     """
#     batchsize = output.shape[0]
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if index is None:
#         index = torch.argmax(output)  # [b]
#     elif type(index) is int:
#         index = torch.ones(batchsize) * index
#     if len(index.shape) == 0:
#         index = torch.ones(batchsize) * index.to(torch.device("cpu"))
#
#     index = torch.unsqueeze(index, 1).to(torch.int64)
#     one_hot = torch.zeros(batchsize, num_classes).scatter_(1, index, 1).to(device)
#     one_hot.requires_grad = True
#     class_vec = torch.mean(one_hot * output)
#
#     return class_vec
#
#
# class encode():
#     def __init__(self,
#                  module,
#                  pthfile):
#         self.module = module
#         self.module.load(pthfile)
#         self.module.eval()
#
#     def __call__(self,
#                  input):
#         out = self.module(input)
#         out.detach_()
#         return out


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    y = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    c_m = confusion_matrix(y, y_pred)
    print(c_m)
