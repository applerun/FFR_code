from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn


# __all__ = []

def plt_res(pltdir,
            res,
            val_db,
            test_db,
            informations: str = None):
    """

    :param pltdir:
    :param res:
        # res = dict(train_acces:list, val_acces:list,  # 训练过程——正确率
        #            train_losses:list, val_losses:list,  # 训练过程——损失函数
        #            best_acc:float, # 验证集最佳正确率
        #            best_epoch:int,  # early-stopping位置

        #            res_test:dict  # 测试集所有指标
        #            res_val:dict,  # 验证集所有指标
            res_val/res_test = dict(
                                    # resacc：float正确率, loss:float
                                    # label2roc:dict 各个label的ROC,
                                    # label2auc:dict 各个label的AUC,
                                    # confusion_matrix:np.ndarray 混淆矩阵
                                    )
        #            cams = cams,  # 梯度加权类激活映射图谱(未实现)
    :param informations:
    :return:
    """
    if informations is None:
        informations = ""
    elif not informations.endswith("_"):
        informations += "_"

    label2name = val_db.label2name()

    plt_loss_acc(pltdir, res, informations)
    plt_res_val(pltdir, res["res_val"], label2name, informations = informations + "val")
    plt_res_val(pltdir, res["res_test"], label2name, informations = informations + "test")
    # plt_cam(pltdir, res["val_cam"], val_db, "val")
    # plt_cam(pltdir, res["test_cam"], test_db, "val")


def heatmap(matrix,
            path,
            labels: list = "auto"):
    cm_fig, cm_ax = plt.subplots()
    seaborn.heatmap(matrix, annot = True, cmap = "Blues", ax = cm_ax, xticklabels = labels, yticklabels = labels)
    cm_ax.set_title('confusion matrix')
    cm_ax.set_xlabel('predict')
    cm_ax.set_ylabel('true')
    cm_fig.savefig(path)
    plt.close(cm_fig)


def plt_loss_acc(
        pltdir,
        res,
        informations = None):
    trainfig, trainax = plt.subplots(1, 2)  # 绘制训练过程图
    trainfig.suptitle('train process' + informations)

    epochs = np.arange(len(res["train_losses"]))

    loss_ax, acc_ax = trainax

    loss_ax.set_title("loss")
    loss_ax.plot(epochs, res["train_losses"], label = "train_loss", color = "red")
    loss_ax.plot(epochs, res["val_losses"], label = "val_loss", color = "blue")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend()

    acc_ax.set_title("accuracy")
    acc_ax.plot(epochs, res["train_acces"], label = "train_accuracy", color = "red")
    acc_ax.plot(epochs, res["val_acces"], label = "val_accuracy", color = "blue")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.legend()
    plt.subplots_adjust(wspace = 0.25)
    trainfig.savefig(os.path.join(pltdir, "train_process.png"))
    plt.close(trainfig)


def plt_res_val(pltdir,
                res,
                label2name,
                informations = None):
    if informations is None:
        informations = ""
    else:
        informations = informations + "_"
    label2roc = res["label2roc"]
    label2auc = res["label2auc"]
    for label in label2roc.keys():
        fpr, tpr, thresholds = label2roc[label]
        auc = label2auc[label]
        roc_fig, roc_ax = plt.subplots()
        roc_fig.suptitle("ROC_curve")
        roc_ax.set_title("auc = {}".format(auc))
        roc_ax.plot(fpr, tpr)
        roc_ax.set_xlabel("fpr")
        roc_ax.set_ylabel("tpr")
        roc_fig.savefig(os.path.join(pltdir, informations + "_" + label2name[label] + "_roc.png"))
        plt.close(roc_fig)
    confusion_matrix = res["confusion_matrix"]
    n = confusion_matrix.shape[0]
    ticks = [None for _ in range(n)]
    for label in label2name.keys():
        name = label2name[label]
        ticks[label] = name
    heatmap(confusion_matrix, informations + "_confusion_matrix.png", labels = ticks)
