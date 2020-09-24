"Utilities for object-level-SLAM map building"

__author__ = 'Zherlock'

import os
import time
import numpy as np
import cv2
import pickle
from .instance import Instance
from .mappoint import MapPoint, Property

def init_mps(mp_path: str):
    """read mappoints in the files
    return a list of map points, and set of int (keyframe id)
    """
    if not os.path.exists(mp_path):
        print('no slam mps results, return.')
        return 1
    cnt = 0
    points = []
    keyframe_set = set()
    last_mp = -1
    with open(mp_path, "r") as f:
        for line in f:
            s = line.strip('\n').split(',')
            # print(s)
            assert 'global' in s[0]
            gl_id = int(s[1])
            m_id = int(s[3])
            x_cor = int(float(s[5]))
            y_cor = int(float(s[7]))
            #print("glid is {}, m_id is {}, x_cor is {}, y_cor is {}".format(gl_id, m_id, x_cor, y_cor))
            if gl_id != last_mp:
                last_mp = gl_id
                p = MapPoint(int(s[1]))
                points.append(p)
                p.info[m_id] = Property()
                p.info[m_id].x = x_cor
                p.info[m_id].y = y_cor
                cnt += 1
            else:
                p.info[m_id] = Property()
                p.info[m_id].x = x_cor
                p.info[m_id].y = y_cor
            keyframe_set.add(m_id)
    print("{} key frames and {} mppoints are envolved".format(
        len(keyframe_set), cnt))
    return points, keyframe_set


def init_mats(mat_path: str):
    """read mat file names in the folder
    save them in a list
    return a list of mat filenames 
    """
    dirs = os.listdir(mat_path)
    mats = sorted(dirs, key=lambda x: float(x[:-4]))  # 所有图片文件名
    assert mats[0] == '00001.png'
    return mats


def readmat(m_id: int, mats: list, mat_path: str  ):
    """fetch the mat using cv2.imread
    """
    return cv2.imread(os.path.join(mat_path + mats[m_id]))


def save_obj(obj, name):
    """save dictionary memo using pickle
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """load dictionary memo using pickle
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def visualize_p(p: MapPoint, mats:list, mat_path:str):
    t_wait = 1300 # 点和点之间的间隔时间
    cnt = 0
    for m_id in p.info.keys():
        im = readmat(m_id, mats, mat_path)
        x = p.info[m_id].x
        y = p.info[m_id].y
        cv2.circle(im, (int(x), int(y)), 3, (0, 0, 255), 2)
        cv2.imshow('test', im)
        cv2.waitKey(t_wait)
        cv2.destroyAllWindows()
        cnt += 1
        if cnt > 0:
            break

def instance_labeling(ins_list:list, points:list):
    """give each map point a instance id
    """
    # 现在看这个函数觉得很奇怪，point[下标为列表下标], ins.mps存储的是point的global_id，这两个应该联系不起来的
    # 的确是联系起来的，point是一个个加入列表的，列表下标刚好和point的global_id一样
    for ins in ins_list:
        for p_id in ins.mps:
            points[p_id].instance_id = ins.id

def save_results(points:list, label_path:str):
    cnt = 0
    # for p in points:
    #    print(p.instance_id)
    with open(os.path.join(label_path, "instance_saving.txt"), "w") as f:
        for p in points:
            if p.instance_id != -1:
                f.write(str(p.global_id))
                f.write(' ')
                f.write(str(p.instance_id))
                f.write('\n')
                cnt += 1
    print('total, ', cnt, ' labeled points')

if __name__ == '__main__':
    pass

