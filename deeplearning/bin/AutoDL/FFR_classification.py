import builtins

import torch
from torch import nn, optim
import visdom
from deeplearning.utils.DataLoader import MultiEpochsDataLoader as DataLoader
import torchvision.transforms as transforms
from deeplearning.utils import iterator, VideoData
from deeplearning.utils.validation import evaluate, evaluate_loss, evaluate_all
from deeplearning.models.BasicModule import BasicModule
from deeplearning.models.RNN.CNN_RNN import BaseModelRNN
from deeplearning.utils.plt_utils import *
from torchvision import models
import time
import csv
import numpy as np
import pysnooper
import warnings

# 环境
projectroot = "../../.."

# 记录
logfile = os.path.join(projectroot, "log", "ffr", "FFR_classification", time.strftime("%Y-%m-%d-%H_%M_%S") + ".txt")
if not os.path.isdir(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))
with open(logfile, "w") as f:
    pass


@pysnooper.snoop(
    logfile,
    prefix = "--*--")
def main_train(
        net: BasicModule,
        train_db,
        val_db,
        test_db,
        device,
        lr = 0.0001,
        batchsz = 4,
        epochs = 100,
        epo_interv = 1,
        numworkers = 4,
        vis = None,
        modelname = None,
        net_save_dir = os.path.join(projectroot, "checkpoints", "Default"),
        criteon = nn.CrossEntropyLoss(),
        verbose = False,
        k = 0,

):  # 训练网络并返回结果
    if vis == None:
        vis = visdom.Visdom()

    if not modelname is None:
        net.set_model_name(modelname)

    # create loaders\
    train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
                              pin_memory = True)
    val_loader = DataLoader(val_db, batch_size = batchsz, num_workers = numworkers)
    test_loader = DataLoader(test_db, batch_size = batchsz, num_workers = numworkers)

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr)

    best_acc, best_epoch = 0, 0
    global_step = 0

    steps = []
    train_losses, val_losses, test_losses = [], [], []
    train_acces, val_acces, test_accses = [], [], []

    vis.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
    vis.line([0], [-1], win = "val_loss_" + str(k), opts = dict(title = "val_loss_" + str(k)))
    vis.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
    vis.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k)))

    if not os.path.isdir(net_save_dir):
        os.makedirs(net_save_dir)
    net_save_path = os.path.join(net_save_dir, net.model_name + ".mdl")

    if not os.path.exists(net_save_path):
        with open(net_save_path, "w", newline = ""):
            pass

    print("start training")
    for epoch in range(epochs):
        loss = iterator.train(net, lr, device, train_loader, criteon, optimizer, epoch, mixed = False,
                              verbose = verbose)
        vis.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
        global_step += 1

        if epoch % epo_interv != 0:
            continue
        net.eval()
        with torch.no_grad():
            val_loss = evaluate_loss(net, val_loader, criteon, device)
            # test_loss = evaluate_loss(net,test_loader,criteon,device)
            train_acc = evaluate(net, train_loader, device)
            val_acc = evaluate(net, val_loader, device)
            # test_acc = evaluate(net, test_loader, device)

            train_acces.append(train_acc)
            val_acces.append(val_acc)
            # test_accses.append(test_acc)
            train_losses.append(loss)
            val_losses.append(val_loss)
            # test_losses.append(test_loss)
            steps.append(global_step)
            vis.line([val_loss], [global_step], win = "val_loss_" + str(k), update = "append")
            vis.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
            vis.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
            # vis.line([test_acc], [global_step], win = "test_acc_" + str(k), update = "append")
            if val_acc >= best_acc and epoch > epochs / 5:
                best_epoch = epoch
                best_acc = val_acc
                net.save(net_save_path)

        # 若需要对每个分类的数据进行单独分析，请完成以下代码
        # if (epoch % epo_interv == 0 or epoch == epochs - 1) and verbose:
        #     sample2acc_train = evaluate_samplewise(net, val_db, device)
        #     sample2acc_val = evaluate_samplewise(net, val_db, device)
        #     sample2acc_test = evaluate_samplewise(net, test_db, device)
        #     batch_plt(
        #         sample2acc_val, global_step, win = "val_acc_each_sample" + str(k),
        #         update = None if global_step <= epo_interv else "append", viz = vis
        #     )
        #     batch_plt(
        #         sample2acc_test, global_step, win = "test_acc_each_sample" + str(k),
        #         update = None if global_step <= epo_interv else "append", viz = vis
        #     )
        #     batch_plt(sample2acc_train, global_step, win = "val_acc_each_sample" + str(k),
        #               update = None if global_step <= epo_interv else "append", viz = vis)

    # 测试集
    net.load(net_save_path)

    # print("best_acc:", best_acc, "best epoch", best_epoch)
    # print("loaded from ckpt!")
    res_test = evaluate_all(net, test_loader, criteon, device)
    res_val = evaluate_all(net, val_loader, criteon, device)
    res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
               train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
               best_acc = best_acc, best_epoch = best_epoch,  # early-stopping位置
               res_test = res_test,  # 测试集所有指标：
               # acc：float正确率, loss:float,
               # label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
               res_val = res_val,  # 验证集所有指标

               )
    return res


def npsv(pltdir, res, val_db, ):  # 结果保存
    """

    :param pltdir:
    :param res:
    :param val_db:
    :return:
    """
    # 保存训练过程
    process = np.array([res["train_acces"], res["val_acces"], res["train_losses"], res["val_losses"]]).T
    np.savetxt(os.path.join(pltdir, "train_process.csv"), process,
               header = "train_acces,val_acces,train_losses,val_losses", delimiter = ",")

    label2roc_val = res["res_val"]["label2roc"]
    label2roc_test = res["res_test"]["label2roc"]

    # 保存混淆矩阵
    np.savetxt(os.path.join(pltdir, "val_confusion_matrix.csv"), res["res_val"]["confusion_matrix"],
               delimiter = ",")
    np.savetxt(os.path.join(pltdir, "test_confusion_matrix.csv"), res["res_test"]["confusion_matrix"],
               delimiter = ",")

    for label in val_db.label2name().keys():
        name = val_db.label2name()[label]
        # 保存ROC
        fpr, tpr, thresholds = label2roc_val[label]
        np.savetxt(os.path.join(pltdir, "val" + "_" + name + "_roc.csv"), np.vstack((fpr, tpr, thresholds))
                   , delimiter = ",")
        fpr, tpr, thresholds = label2roc_test[label]
        np.savetxt(os.path.join(pltdir, "test" + "_" + name + "_roc.csv"), np.vstack((fpr, tpr, thresholds))
                   , delimiter = ",")

        # 保存CAM
        try:
            val_cam = res["val_cam"][label]
            test_cam = res["test_cam"][label]
            xs = np.linspace(val_db.xs[0], val_db.xs[-1], val_cam.shape[-1])
            xs = np.expand_dims(xs, axis = 0)
            if not os.path.exists(os.path.join(pltdir, "cam")):
                os.makedirs(os.path.join(pltdir, "cam"))
            np.savetxt(os.path.join(pltdir, "cam", "val_cam_" + name + "_activated.csv"), np.vstack((xs, val_cam)),
                       delimiter = ",")
            np.savetxt(os.path.join(pltdir, "cam", "test_cam_" + name + "_activated.csv"), np.vstack((xs, test_cam)),
                       delimiter = ",")
        except:
            warnings.warn("cam output failed")
    return


def main():  # 训练各种类型的网络并保存结果
    k_split = 6
    n_iter = 1
    basemodel = models.resnet18(pretrained = False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelname = "RNN"

    train_transformer = transforms.Compose([
        # transforms.RandomHorizontalFlip(p = 0.5),
        # transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
        transforms.ToTensor(),
    ])
    test_transformer = transforms.ToTensor()
    dataroot = os.path.join("..", "..", "..", "FFR_data", "CAG_raw_jpg")
    vis = visdom.Visdom()
    db_cfg = dict(
        dataroot = dataroot, k_split = k_split,
        t_v_t = [1.0, 0.0, 0.0],
        timesteps = 70,
    )
    train_cfg = dict(
        device = device,
        lr = 0.0001,
        vis = vis,
        modelname = modelname,
        batchsz = 4,

    )

    # 模型配置
    CNN_Model = BaseModelRNN
    cnn_model_cfg = dict(
        num_classes = 5, dr_rate = 0.2, num_hiddens = 100,
        basenet = None,
        rnn = nn.LSTM, num_features = 512,
        rnn_num_layers = 1,
    )
    # dict(
    #     num_classes = 2, dr_rate = 0.2, num_hiddens = 100, baseModel = None, rnn = None, num_features = 100,
    #     rnn_num_layers = 1,
    # )

    # 保存配置
    recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
    recorddir = os.path.join(projectroot, "results", "ffr", recorddir)
    if not os.path.isdir(recorddir):
        os.makedirs(recorddir)
        train8val(k_split, CNN_Model, recorddir, db_cfg, train_transformer, test_transformer, n_iter, train_cfg,
                  cnn_model_cfg, device)


def train8val(k_split, CNN_Model, recorddir, db_cfg, train_transformer, test_transformer, n_iter, train_cfg,
              model_cfg, device):  # 将每个网络训练结果保存
    if not os.path.isdir(recorddir):
        os.makedirs(recorddir)
    recordfile = recorddir + ".csv"  # 记录训练的配置和结果

    f = open(recordfile, "w", newline = "")
    writer = csv.writer(f)
    f.write(db_cfg.__str__() + "\n")
    f.write(train_cfg.__str__() + "\n")
    writer.writerow(["n", "k", "best_acc", "test_acc", "best_epoch", "val_AUC", "test_AUC"])
    conf_m_v = np.zeros((model_cfg["num_classes"], model_cfg["num_classes"]))
    conf_m_t = np.zeros((model_cfg["num_classes"], model_cfg["num_classes"]))
    bestaccs, testaccs, bepochs, vaucs, taucs = [], [], [], [], []
    i = 0
    assert n_iter > 0 and k_split > 1
    for n in range(n_iter):
        for k in range(k_split):
            sfpath = "Raman_" + str(n) + ".csv"
            train_db = VideoData.FFRDataset(**db_cfg, transform = train_transformer, k = k, sfpath = sfpath)
            val_db = VideoData.FFRDataset(**db_cfg, mode = "val", transform = test_transformer, k = k, sfpath = sfpath)
            assert len(train_db) > 0 and len(val_db) > 0, "empty database"

            # 创建网络
            net = CNN_Model(**model_cfg).to(device = device)
            res = main_train(
                net,
                train_db,
                val_db,
                val_db,
                **train_cfg,
            )
            pltdir = os.path.join(recorddir, "n-{}-k-{}".format(n, k))
            net.save(os.path.join(pltdir, net.model_name + ".mdl"))
            if not os.path.isdir(pltdir):
                os.makedirs(pltdir)

            b, t, be, auc_val, auc_test = res["best_acc"], res["res_test"]["acc"], res["best_epoch"], \
                                          np.mean(list(res["res_val"]["label2auc"].values())), \
                                          np.mean(list(res["res_test"]["label2auc"].values()))
            writer.writerow([n, k, b, t, be, auc_val, auc_test])

            bestaccs.append(b)
            testaccs.append(t)
            bepochs.append(be)
            vaucs.append(auc_val)
            taucs.append(auc_test)
            i += 1
            print(i, "/", n_iter * k_split)

            plt_res(pltdir, res, val_db, val_db, informations = None)
            npsv(pltdir, res, val_db, )
            conf_m_v += res["res_val"]["confusion_matrix"]
            conf_m_t += res["res_test"]["confusion_matrix"]
    np.savetxt(os.path.join(recorddir, "test_confusion_matrix.csv"), conf_m_v, delimiter = ",")
    np.savetxt(os.path.join(recorddir, "val_confusion_matrix.csv"), conf_m_t, delimiter = ",")
    heatmap(conf_m_t, os.path.join(recorddir, "test_confusion_matrix.png"), labels = list(val_db.label2name().values()))
    heatmap(conf_m_v, os.path.join(recorddir, "val_confusion_matrix.png"), labels = list(val_db.label2name().values()))

    # train_db.shufflecsv()

    ba = np.mean(np.array(bestaccs)).__str__() + "+-" + np.std(np.array(bestaccs)).__str__()
    ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
    bea = np.mean(np.array(bepochs)).__str__() + "+-" + np.std(np.array(bepochs)).__str__()
    auc_av = np.mean(np.array(vaucs)).__str__() + "+-" + np.std(np.array(vaucs)).__str__()
    auc_at = np.mean(np.array(taucs)).__str__() + "+-" + np.std(np.array(taucs)).__str__()
    writer.writerow(["mean", "std", ba, ta, bea, auc_av, auc_at])
    f.close()

    print("best acc:", ba)
    print("test acc", ta)
    print("best epochs", bea)


if __name__ == '__main__':
    main()
