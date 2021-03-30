from os import path
from acoular import __file__ as bpath, td_dir, MicGeom, WNoiseGenerator, PointSource, Mixer, WriteH5
import numpy as np
import os
import h5py

sfreq = 51200  # 采样频率
duration = 1  # 采样时间
nsamples = duration * sfreq  # 采样点个数
#micgeofile = path.join(path.split(bpath)[0], 'xml', 'MyMicArray_56.xml')  # 麦克风阵列的位置数据
micgeofile = '/home3/zengwh/VoiceRec/data/compute/MyMicArray_56.xml'
num_single_sound_data = 100  # 单声源训练集数目
m = MicGeom(from_file=micgeofile)

# 设置声源位置信息和声压强度
# n2 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7)
# n3 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5)
for i in range(num_single_sound_data):
    print(i)
    [Rms] = np.around(np.random.random(1), 2)
    n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, rms=Rms)
    # 随机产生0-1的数作为声源坐标并进行四舍五入操作
    [source_x, source_y] = np.dot(np.around(np.random.random(2), 2),3)-1.5
    p1 = PointSource(signal=n1, mpos=m, loc=(source_x, source_y, 2.5))
    # p2 = PointSource(signal=n2, mpos=m,  loc=(0.15,0,0.3))
    # p3 = PointSource(signal=n3, mpos=m,  loc=(0,0.1,0.3))
    # p = Mixer(source = p1, sources = [p2])
    p = Mixer(source=p1)
    # 保存文件名
    os.makedirs('/home3/zengwh/VoiceRec/data/compute/100000_data/Val/one_source',exist_ok=True)
    h5savefile = '/home3/zengwh/VoiceRec/data/compute/100000_data/Val/one_source/x_{:.2f}_y_{:.2f}_rms_{:.2f}_sources.h5'.format(source_x, source_y, Rms)
    wh5 = WriteH5(source=p, name=h5savefile)
    #print(wh5)
    wh5.save()
    h5 = h5py.File(h5savefile)
    h5 =h5['time_data'][::50,:]
    os.system("rm {}".format(h5savefile))
    np.save(h5savefile[:-2]+'npy',h5)
