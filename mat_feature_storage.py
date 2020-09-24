"""use yolov5s to extract features of submats of certain instance
store in folder feature_memo in npy file
"""

'''
子图里存在
1.完全错误的
2.类别包含的，同一张图的子图，连续被识别为键盘，laptop，tv，应该选择laptop吧
在read_sub_mat中用clean函数解决
'''

__author__ = 'zherlock'

import sys
sys.path.append(r'/home/zherlock/InstanceDetection/yolov5_original/')

#print(sys.path)

import os
import numpy as np
import argparse
import time
import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from read_sub_mat import * 
import pickle


def new_letter_box(img):
    """modify the mat size to let w%32==0 and h%32==0
    """
    print('img shaps is ', img.shape) # (480,640,3)
    w, h = img.shape[1], img.shape[0]
    print('orginal w is ', w, ' h is ', h)
    new_w, new_h = w, h
    if w % 32 != 0:
        new_w = w + 32 - w % 32
    if h % 32 != 0:
        new_h = h + 32 - w % 32
    print("new_w is ", new_w, " and new_h is ", new_h)
    dw = new_w - w
    dh = new_h - h
    dw = dw / 2
    dh = dh / 2
    print("dw is ", dw, " and dh is ", dh)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


if __name__ == '__main__':
    ins_path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/label/InsList.npy'
    ins_list = np.load(ins_path).tolist()
    mat_path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/rgb/'
    dirs = os.listdir(mat_path)
    #所有图片文件名
    global mats
    mats = sorted(dirs, key = lambda x: float(x[:-4]))
    assert mats[0] == '00001.png'

    mat_id_set = set()  # 存储需要yolo计算的mat_id.
    for ins in ins_list:
        ins.mat_set  = clean(ins.mat_set)
        #print(len(ins.mat_set))
        for submat_info in ins.mat_set:
            #print(submat_info)
            m_id = submat_info[0]
            if m_id in mat_id_set:
                pass
                #print('m_id ', m_id, ' duplicated')
            mat_id_set.add(m_id)

    print('total {} mats are envolved.'.format(len(mat_id_set)))
    #sys.exit(0)

    device = torch_utils.select_device('cpu')
    weights = '/home/zherlock/InstanceDetection/yolov5_original/weights/yolov5s.pt'
    model = torch.load(weights, map_location=device)['model'].float().eval()  # load FP32 model

    '''
    name = 'feature_memo'
    cnt = 0
    with open(name + '.pkl', 'rb') as f:
        feature_memo = pickle.load(f)
    for key, value in feature_memo.items():
        print('cnt is ', cnt)
        cnt += 1
        print('mid is ', key, ' features is ', value.shape)
    sys.exit(0)
    '''
    folder = 'feature_memo'
    if not os.path.exists(folder):
        os.makedirs(folder)
    #batch = 20
    #feature_memo = {}
    for i, m_id in enumerate(mat_id_set):
        #break
        img = readmat(mat_path, mats, m_id)
        #img = new_letter_box(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        start = time.time()
        pred = model(img, augment=False) # list of features, pred[0] shape [1, 255, 60, 80]
        tcost = time.time() - start
        print(i, ' th mat takes ', tcost, ' secs')
        if i == 0:
            t_baseline = tcost
        if tcost / t_baseline > 5:
            print('here we met worst case.')
            break
        array = pred[0].detach().numpy()
        name = os.path.join(folder, str(m_id)+'.npy')
        np.save(name, array)
        if i > 5:
            pass
            #break

        #print(pred[0].shape)
        #print(pred[1].shape)
        #print(pred[2].shape)
        #feature_memo[m_id] = pred[0]
        #if i % batch == batch - 1:
        #    name = 'feature_memo_' + str(int(i / batch))
        #    with open(name + '.pkl', 'wb') as f:
        #        pickle.dump(feature_memo, f, pickle.HIGHEST_PROTOCOL)
        #    feature_memo = {}
        #break
        

'''
path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/submats/'
index_o = 0
while(os.path.exists(os.path.join(path, str(index_o)))):
    print('get in')
    sub_path = os.path.join(path, str(index_o))
    index_i = 0
    while(os.path.exists(os.path.join(path, str(index_o), str(index_i)+'.npy'))):
        submat = np.load(os.path.join(path, str(index_o), str(index_i)+'.npy'))
        print('shape is ', submat.shape)
        index_i += 1
    index_o += 1
    if index_o > 1:
        break
'''