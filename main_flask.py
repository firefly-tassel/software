# 定义全局的app路由
# 然后将其他模块由app装饰后的函数导入，flask即可识别所有的请求入口
import shutil
import zipfile
from sqlalchemy import create_engine
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import traceback
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pandas as pd
import json
import scipy.io as sio
import matplotlib
import numpy as np
import os

from SelfSupModels.runner_function import selfallargs
from SelfSupModels.selfmain import mainrunnerself_flask, ftself_flask
from SelfSupModels.selftest import selftest_flask, selftestall_flask
from NonSupModels.runner_function import nonallargs
from NonSupModels.nonmain import mainrunnernon_flask, ftnon_flask
from NonSupModels.nontest import nontestall_flask
from OtherModels.tstcc.tstccmain import otherallargs, runner_flask
from utils import get_host_ip

matplotlib.use('Agg')

executor = ThreadPoolExecutor(1000)

# 配置全局app
app = Flask(__name__, static_folder='./static')
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')
name_space = '/ws'

# 初始化数据库连接
# 按实际情况依次填写MySQL的用户名、密码、IP地址、端口、数据库名
con = create_engine('mysql+pymysql://root:17870391037phz@localhost:3306/vuedb')

# 协议域名端口
port = 5000
# url = f'http://localhost:{port}'
url = f'http://{get_host_ip()}:{port}'

data_set = {}  # 整个数据集
data_train = None  # 训练数据
gt_train = None  # 训练数据标签
model_train = {}  # 训练模型
model_test = {}  # 测试模型
lines, losses, accuracy = {}, {}, {}  # 训练日志数据

# 处理前文件路径
static_image_path = '/static/images/'
static_npy_path = '/static/npy/'
static_pretrain_weight_path = '/static/pretrain_weight/'
static_weight_path = '/static/weight/'
static_waveform_path = '/static/waveform/'
# 处理后文件路径


def parse_path(origin_path):
    return './' + origin_path[len(url) + 1:]


@socketio.on('connect', namespace=name_space)
def connected_msg():
    print('client connected')


@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    print('client disconnected')


# 在这里写对应的http接口函数
@app.route('/', methods=['GET', 'POST'])
def index():
    """主页返回"""
    return 'I have receive your request !'


@app.route('/api/getIP', methods=['GET', 'POST'])
def getIP():
    """查询本机ip地址"""
    try:
        ip = get_host_ip()
        return json.dumps({'code': 1, 'msg': '获取成功', 'ip': ip})
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': str(e)})


@app.route('/api/upload', methods=['GET', 'POST'])
def upload():
    """上传文件"""
    global url
    try:
        file = request.files.get("file")
        if file.filename.endswith('.csv'):
            path = './static/excel'
            if not os.path.exists(path):
                os.makedirs(path)
            file.save(os.path.join(path, file.filename))
            return json.dumps({'code': 1, 'msg': '上传成功'})
        elif file.filename.endswith('.mp4'):
            originVideo = url + '/static/videos/' + str(file.filename)
            path = './static/videos'
            if not os.path.exists(path):
                os.makedirs(path)
            file.save(os.path.join(path, file.filename))
            return json.dumps({'code': 1, 'msg': '上传成功', 'originVideo': originVideo})
        elif file.filename.endswith('.npy'):
            path = static_npy_path[1:]
            if not os.path.exists(path):
                os.makedirs(path)
            file.save(os.path.join(path, file.filename))
            return json.dumps({'code': 1, 'msg': '上传成功'})
        else:
            originImg = url + static_image_path + str(file.filename)
            path = static_image_path[1:]
            if not os.path.exists(path):
                os.makedirs(path)
            file.save(os.path.join(path, file.filename))
            return json.dumps({'code': 1, 'msg': '上传成功', 'originImg': originImg})
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': '上传失败，错误原因请查看服务器'})


@app.route('/api/get_pretrain_weight', methods=['GET', 'POST'])
def get_pretrain_weight():
    """"获取预训练权重"""
    try:
        weight_list = []
        for file in sorted(os.listdir(static_pretrain_weight_path[1:])):
            if file.endswith('.pt'):
                weight_list.append(file.split('.pt')[0])
        return json.dumps({'code': 1, 'msg': '获取成功！', 'weight_list': weight_list})
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': str(e)})
    

@app.route('/api/get_waveform', methods=['GET', 'POST'])
def get_waveform():
    """"获取所有信号波形图"""
    try:
        waveform_list = []
        for file in sorted(os.listdir(static_waveform_path[1:])):
            if file.endswith('.png'):
                waveform_list.append(f'{url}{static_waveform_path}{file}')
        return json.dumps({'code': 1, 'msg': '获取成功！', 'waveform_list': waveform_list})
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': str(e)})


@app.route('/api/pretrain', methods=['GET', 'POST'])
def pretrain():
    """"预训练"""
    try:
        inf = request.get_json()
        print(inf)

        task = inf['task']

        framework = inf['framework']
        backbone = inf['backbone']
        n_epoch = inf['n_epoch']

        traindatapath = 'X_train.npy'
        traingtpath = 'Y_train.npy'
        valdatapath = 'X_train.npy'
        valgtpath = 'Y_train.npy'
        testdatapath = 'X_train.npy'
        testgtpath = 'Y_train.npy'

        if task == 1:
            args = selfallargs().parse_args() if framework!= 'TSTCC' else otherallargs().parse_args()
            args.framework = framework
            args.backbone = backbone
            args.n_epoch = n_epoch
            args.traindatapath = traindatapath
            args.traingtpath = traingtpath
            args.valdatapath = valdatapath
            args.valgtpath = valgtpath
            args.testdatapath = testdatapath
            args.testgtpath = testgtpath
            args.socketio = socketio
            args.name_space = name_space
            args.executor = executor

            if args.framework != 'TSTCC':
                mif, maf = mainrunnerself_flask(args)
            else:
                runner_flask(args)
        elif task == 2:
            args = nonallargs().parse_args()
            args.framework = framework
            args.backbone = backbone
            args.n_epoch = n_epoch
            args.traindatapath = traindatapath
            args.traingtpath = traingtpath
            args.valdatapath = valdatapath
            args.valgtpath = valgtpath
            args.testdatapath = testdatapath
            args.testgtpath = testgtpath
            args.socketio = socketio
            args.name_space = name_space
            args.executor = executor

            mif, maf = mainrunnernon_flask(args)

        return json.dumps({
            'code': 1,
            'msg': '训练结束！',
        })
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps(
            {'code': 0, 'msg': '训练失败！', 'record': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


@app.route('/api/finetune', methods=['GET', 'POST'])
def finetune():
    """"微调"""
    try:
        inf = request.get_json()
        print(inf)

        task = inf['task']  # 1-个体识别 2-信号分选

        framework = inf['framework']  # [byol, simsiam, simclr, TSTCC]
        backbone = inf['backbone']  # [resnet, Transformer]
        finetune_epoch = inf['finetune_epoch']

        # pretrain_weight = inf['pretrain_weight']  # 预训练权重，传文件名
        # pretrain_weight = 'results/pretrain_try_scheduler_simclr_pretrain_data57_eps5_lr0.0001_bs8_aug1na_aug2na_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm4.pt'
        pretrain_weight = 'results/pretrain_try_scheduler_simclr_pretrain_data57_eps5_lr0.0001_bs32_aug1na_aug2na_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm4.pt'

        traindatapath = 'X_train.npy'
        traingtpath = 'Y_train.npy'
        valdatapath = 'X_train.npy'
        valgtpath = 'Y_train.npy'
        testdatapath = 'X_train.npy'
        testgtpath = 'Y_train.npy'

        if task == 1:
            args = selfallargs().parse_args() if framework!= 'TSTCC' else otherallargs().parse_args()
            args.framework = framework
            args.backbone = backbone
            args.finetune_epoch = finetune_epoch
            args.pretrain_weight = pretrain_weight
            args.traindatapath = traindatapath
            args.traingtpath = traingtpath
            args.valdatapath = valdatapath
            args.valgtpath = valgtpath
            args.testdatapath = testdatapath
            args.testgtpath = testgtpath
            args.socketio = socketio
            args.name_space = name_space
            args.executor = executor

            if args.framework != 'TSTCC':
                mif, maf = ftself_flask(args)

        elif task == 2:
            args = nonallargs().parse_args()
            args.framework = framework
            args.backbone = backbone
            args.finetune_epoch = finetune_epoch
            args.pretrain_weight = pretrain_weight
            args.traindatapath = traindatapath
            args.traingtpath = traingtpath
            args.valdatapath = valdatapath
            args.valgtpath = valgtpath
            args.testdatapath = testdatapath
            args.testgtpath = testgtpath
            args.socketio = socketio
            args.name_space = name_space
            args.executor = executor

            mif, maf = ftnon_flask(args)

        return json.dumps({
            'code': 1,
            'msg': '微调结束！',
        })
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps(
            {'code': 0, 'msg': '微调失败！', 'record': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


@app.route('/api/individual_recog', methods=['GET', 'POST'])
def individual_recog():
    """个体识别"""
    try:
        inf = request.get_json()
        print(inf)

        mode = inf['mode']  # 1-批量识别 2-单个识别

        backbone = inf['backbone']  # [resnet, Transformer]
        framework = inf['framework']  # [byol, simsiam, simclr, TSTCC]

        gt_path = 'Y_train.npy'  # 标签文件
        # 权重文件，传文件名
        weight_path = 'results/lincls_try_scheduler_simclr_pretrain_data57_eps5_lr0.0001_bs8_aug1na_aug2na_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm1.pt'

        args = selfallargs().parse_args()
        args.backbone = backbone
        args.framework = framework

        if mode == 1:
            # 批量，传信号索引列表
            X_train = np.load('X_train.npy')
            testdatas = X_train
            results = selftestall_flask(testdatas, weight_path, args)
            gt = np.load(gt_path)
            acc = (np.array(results) == gt).sum() / gt.shape[0]
            # self.ui.AccLabel.setText('Acc:' + str(100 * round(acc, 4)) + '%')
            print('Acc:' + str(100 * round(acc, 4)) + '%')
        elif mode == 2:
            # 单个，传信号索引
            X_train = np.load('X_train.npy')
            testdata = np.expand_dims(X_train[0], axis=0)
            output = selftest_flask(testdata, weight_path, args)
            print(len(output), output.index(max(output)), max(output))
        return json.dumps({
            'code': 1,
            'msg': '个体识别成功！',
        })
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': str(e)})
    

@app.route('/api/signal_sorting', methods=['GET', 'POST'])
def signal_sorting():
    """信号分选"""
    try:
        inf = request.get_json()
        print(inf)

        X_train = np.load('X_train.npy')
        testdatas = X_train

        backbone = inf['backbone']  # [resnet, Transformer]
        framework = inf['framework']  # [byol, simsiam, simclr, TSTCC]

        gt_path = 'Y_train.npy'  # 标签文件
        # 权重文件，传文件名
        weight_path = 'results/lincls_try_scheduler_simclr_pretrain_data57_eps5_lr0.0001_bs32_aug1na_aug2na_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm1.pt'

        args = nonallargs().parse_args()
        args.backbone = backbone
        args.framework = framework
        acc = nontestall_flask(testdatas, gt_path, weight_path, args)
        return json.dumps({
            'code': 1,
            'msg': '信号分选成功！',
        })
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)
        return json.dumps({'code': 0, 'msg': str(e)})
    

def main():
    # 启动web服务器
    # app.run(host='0.0.0.0', port=port, debug=True)
    socketio.run(app, host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    main()
