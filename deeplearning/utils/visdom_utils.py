import os
import sys
import time

import requests
import visdom
import webbrowser

coderoot = os.path.dirname(__file__)
for i in range(2):
    coderoot = os.path.dirname(coderoot)

projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )
sys.path.append(coderoot)


def startVisdomServer(url = "http://localhost:8097"):
    try:
        response = requests.get(url, timeout = 5).status_code
        if response != 200:

            os.system("start python -m visdom.server")  # 启动服务器
            time.sleep(10)
            print("server started!")
            webbrowser.open(url)
        else:
            print("server aready started,url = ", url)
        return
    finally:
        os.system("start python -m visdom.server")  # 启动服务器
        time.sleep(2)
        print("server started!")
        webbrowser.open(url)
        return


def batch_plt(name2value: dict,
              x,
              win = 'batch_plt_vis_default',
              update = None,
              viz: visdom.Visdom = None,
              opts: dict = None,
              name2opts: dict = None):
    # assert all([type(k) == str for k in name2value.keys()]), "all names must be str,but got {}".format(
    #     [type(k) for k in name2value.keys()])
    # assert all([type(k) == (float or torch.Tensor or numpy.ndarray) for k in
    #             name2value.values()]), "all values must be float or Tensor,but got {}".format(
    #     [type(k) for k in name2value.values()])
    if viz == None:
        viz = visdom.Visdom()
    if opts is None:
        opts = dict(title = win, showlegend = True)
    for k in sorted(name2value.keys()):
        viz.line(Y = [name2value[k]], X = [x], win = win, update = update, name = str(k) if type(k) != str else k,
                 opts = opts if name2opts is None else name2opts[k])
    return


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from deeplearning.utils import VideoData

    startVisdomServer()
    tvt = [0.8, 0.2, 0]

    dataname = "data_AST"

    train_db = VideoData.FFRDataset(os.path.join(dataroot, dataname), mode = "train", backEnd = "-.csv", t_v_t = tvt)
    train_loader = DataLoader(train_db, batch_size = 16, shuffle = True, num_workers = 0)
