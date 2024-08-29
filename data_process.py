import os

import pywt
import scipy.fftpack as sf

import numpy as np

from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from multiprocessing import Lock

lock = Lock()

def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(sf.fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result


def dataWave(data, name):
    try:
        lock.acquire()
        if not os.path.exists('./static/waveform/'):
            os.makedirs('./static/waveform/')
        if 'WT' in name:
            data = data[0]
        plt.figure(f'{name}_waveform')
        plt.plot(data)
        plt.xlabel('采样点', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.savefig(f"./static/waveform/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def spectrogram(data, name):
    try:
        lock.acquire()
        # 创建保存频谱图的文件夹，画出信号的频谱图并保存为png文件
        if not os.path.exists('./static/spectrogram/'):
            os.makedirs('./static/spectrogram/')
        if 'WT' in name:
            data = data[0]
        x, Spectrogram = FFT(1, data)
        plt.figure(f'{name}_spectrogram')
        plt.plot(Spectrogram)
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅度', fontsize=12)
        plt.savefig(f"./static/spectrogram/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def stft(data, name):
    try:
        lock.acquire()
        # 创建保存时频图的文件夹，画出信号的时频图并保存为png文件
        if not os.path.exists('./static/stft/'):
            os.makedirs('./static/stft/')
        if 'WT' in name:
            data = data[0]
        fre, t, zxx = signal.stft(data, nperseg=16)  # 短时傅里叶变换
        plt.figure(f'{name}_stft')
        plt.pcolormesh(t, fre, np.abs(zxx), shading='auto')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('频率 (Hz)', fontsize=12)
        plt.savefig(f"./static/stft/{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def phaseGram(data, name):
    try:
        lock.acquire()
        # 保存时相图 png
        if not os.path.exists("./static/phase/"):
            os.makedirs("./static/phase/")
        if 'WT' in name:
            data = data[0]
        plt.figure(f'{name}_phase')
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.savefig(f"./static/phase/{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(str(e))
        print(e.__traceback__.tb_lineno)
    finally:
        lock.release()


def plotFigure(filepath, name):
    # 加载数据
    if name.endswith(".dat") or name.endswith(".DAT"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif name.endswith(".pls") or name.endswith(".PLS"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif name.endswith(".mat") or name.endswith(".MAT"):
        data = sio.loadmat(filepath)['data'].reshape(-1)
    else:
        data = np.fromfile(filepath, dtype=np.int16)
    # print(X.shape)
    # X1 = []
    # length = 500
    # for i in range(int(X.shape[0] / int(length))):  # 按length将数据切开
    #     X1.append(X[i * int(length):int(length) * (i + 1)])
    # data = np.array(X1[0])
    # data = np.array(X[0])
    dataWave(data, name)
    spectrogram(data, name)
    stft(data, name)
    phaseGram(data, name)


def eliminateOutlier(filepath):
    """1.异常值剔除"""
    # 加载数据
    # X = np.fromfile(filepath, dtype=np.int16)
    if filepath.endswith(".dat") or filepath.endswith(".DAT"):
        X = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".pls") or filepath.endswith(".PLS"):
        X = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".mat") or filepath.endswith(".MAT"):
        X = sio.loadmat(filepath)['data'].reshape(-1)
    else:
        X = np.fromfile(filepath, dtype=np.int16)
    # X1 = []
    # length = 500
    # for i in range(int(X.shape[0] / int(length))):  # 按length将数据切开
    #     X1.append(X[i * int(length):int(length) * (i + 1)])
    # data = np.array(X1[0])
    # data[250] = 20000
    data = np.array(X)

    # 异常值剔除
    from sklearn.neighbors import LocalOutlierFactor

    predictions = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(data.reshape(-1, 1)).predict(data.reshape(-1, 1))
    data2 = data[predictions == 1]
    dataWave(data2, 'eliminateOutlier_dataWave')
    spectrogram(data2, 'eliminateOutlier_spectrogram')
    stft(data2, 'eliminateOutlier_stft')
    phaseGram(data2, 'eliminateOutlier_phaseGram')

    # if not os.path.exists('./static/waveform/'):
    #     os.makedirs('./static/waveform/')
    # fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # axes[0].plot(data)
    # axes[1].plot(data2)
    # plt.tight_layout()
    # plt.savefig(f"./static/waveform/compare.png", dpi=300, bbox_inches='tight')
    # plt.close()


def smooth(filepath):
    """2.数据平滑"""
    # 加载数据
    if filepath.endswith(".dat") or filepath.endswith(".DAT"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".pls") or filepath.endswith(".PLS"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".mat") or filepath.endswith(".MAT"):
        data = sio.loadmat(filepath)['data'].reshape(-1)
    else:
        data = np.fromfile(filepath, dtype=np.int16)

    # 数据平滑
    # 使用移动平均滤波器平滑数据
    wnd_size = 10
    wnd = np.ones(wnd_size) / wnd_size
    smooth_data = np.convolve(data, wnd, mode="same")

    # 绘制图表
    dataWave(smooth_data, "smooth_dataWave")
    spectrogram(smooth_data, "smooth_spectrogram")
    stft(smooth_data, "smooth_stft")
    phaseGram(smooth_data, "smooth_phaseGram")


def normalize(filepath):
    """3.数据归一化"""
    # 加载数据
    if filepath.endswith(".dat") or filepath.endswith(".DAT"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".pls") or filepath.endswith(".PLS"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".mat") or filepath.endswith(".MAT"):
        data = sio.loadmat(filepath)['data'].reshape(-1)
    else:
        data = np.fromfile(filepath, dtype=np.int16)

    # 归一化
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    # 绘制图表
    dataWave(normalized_data, "normalized_dataWave")
    spectrogram(normalized_data, "normalized_spectrogram")
    stft(normalized_data, "normalized_stft")
    phaseGram(normalized_data, "normalized_phaseGram")


def WT(filepath):
    """4.小波变换"""
    try:
        # 加载数据
        if filepath.endswith(".dat") or filepath.endswith(".DAT"):
            data = np.fromfile(filepath, dtype=np.int16)
        elif filepath.endswith(".pls") or filepath.endswith(".PLS"):
            data = np.fromfile(filepath, dtype=np.int16)
        elif filepath.endswith(".mat") or filepath.endswith(".MAT"):
            data = sio.loadmat(filepath)['data'].reshape(-1)
        else:
            data = np.fromfile(filepath, dtype=np.int16)

        # 小波变换
        WT_data, _ = pywt.cwt(
            data, scales=np.arange(1, 64), wavelet="cmor3-3", sampling_period=1.0
        )

        # 绘制图表
        dataWave(WT_data, "WT_dataWave")
        spectrogram(WT_data, "WT_spectrogram")
        stft(WT_data, "WT_stft")
        phaseGram(WT_data, "WT_phaseGram")
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)


def signalFilter(filepath):
    """5.信号滤波"""

    # 加载数据
    # input = np.fromfile(filepath, dtype=np.int16)
    # data = np.array(input)
    if filepath.endswith(".dat") or filepath.endswith(".DAT"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".pls") or filepath.endswith(".PLS"):
        data = np.fromfile(filepath, dtype=np.int16)
    elif filepath.endswith(".mat") or filepath.endswith(".MAT"):
        data = sio.loadmat(filepath)['data'].reshape(-1)
    else:
        data = np.fromfile(filepath, dtype=np.int16)

    # 带通滤波
    Fs = 400000000  # Hz
    # 其中第一个4表示阶数  []里面的分别表示滤波的下限和上限
    b, a = signal.butter(4, [4 / (Fs / 2), 110000000 / (Fs / 2)], "bandpass")

    # 对上述数据进行带通滤波
    filtered_data = signal.filtfilt(b, a, data, axis=-1)

    # 绘制图表
    dataWave(filtered_data, "filtered_dataWave")
    spectrogram(filtered_data, "filtered_spectrogram")
    stft(filtered_data, "filtered_stft")
    phaseGram(filtered_data, "filtered_phaseGram")


def plot_wave(sample, name):
    plt.figure(figsize=(20, 10))
    plt.plot(sample[0, :], color='red')
    plt.plot(sample[1, :], color='blue')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(f"./static/waveform/{name}.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.close()


if __name__ == '__main__':
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # eliminateOutlier('./data/16QAM.dat')
    # plotFigure('D:\\phz\\deeplearning\\VUE_software\\test_data_mat\\00-Radar00-100000000-2000000-CW--1-0-400000000'
    #            '-9410000000-2021_04_24_802.mat', '1')
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    print(X_train.shape, Y_train.shape)
    print(len(Y_train[Y_train == 0]))
    for i, data in enumerate(X_train):
        plot_wave(data, f'signal_{i}')

