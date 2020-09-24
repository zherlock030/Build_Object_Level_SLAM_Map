"""try to read sub mat from InsList.npy
store them into different folder in npy and png file.
需要对保存的mat_set做清洗，一张图内被重复识别的情况，class出错的情况。
是不是可以假设，最多票的class是正确的，在这个基础上，重复识别，以正确的class为准，如果出现两个同样class，以面积大的为准。
"""

__author__ = 'zherlock'

import label_pkg
import argparse
import os
import sys
import numpy as np
import cv2


def readmat(mat_path:str, mats:list, m_id:int):
    return cv2.imread(os.path.join(mat_path, mats[m_id]))


def duplicate(mset:set):
    mid_set = set()  # 存储需要yolo计算的mat_id.
    for submat_info in mset:
        #print(submat_info)
        m_id = submat_info[0]
        if m_id in mid_set:
            print('m_id ', m_id, ' duplicated')
            return True
            #pass
        mid_set.add(m_id)
    return False

def area(xyxy:tuple):
    return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

def clean(mset:set):
    """delete false submat in mat_set
    if the classname is not the voting one, delete it. If one mat appears many times, choose the best 
    one by classname and area
    """
    name_dic = {}
    mid_dic = {}
    for submat_info in mset:
        m_id, xyxy, classname = submat_info
        if classname in name_dic:
            name_dic[classname] += 1
        else:
            name_dic[classname] = 1
        if m_id in mid_dic:
            if area(xyxy) > mid_dic[m_id]:
                mid_dic[m_id] = area(xyxy)
        else:
            mid_dic[m_id] = area(xyxy)
    main_class = None
    main_vote = 0
    for key, value in name_dic.items():
        if value > main_vote:
            main_vote = value
            main_class = key
    #print('main class of this instance is ', main_class)

    clean_set = set()
    for submat_info in mset:
        if submat_info[2] == main_class:
            if area(submat_info[1]) == mid_dic[submat_info[0]]:
                clean_set.add(submat_info)
    #print('original set size is ', len(mset), ' after cleaning is ', len(clean_set))
    assert not duplicate(clean_set)
    return clean_set


if __name__ == '__main__':

    ins_path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/label/InsList.npy'
    ins_list = np.load(ins_path).tolist()

    mat_path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/rgb/'
    dirs = os.listdir(mat_path)
    #所有图片文件名
    global mats
    mats = sorted(dirs, key = lambda x: float(x[:-4]))
    assert mats[0] == '00001.png'

    folder = '/home/zherlock/SLAM/build_my_dataset/zt301/0/submats'
    save_options = ['npy', 'png']
    save = save_options[1]  # 选择保存png还是npy
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, ins in enumerate(ins_list):
        ins.mat_set  = clean(ins.mat_set)
        if i > 4:
            pass
            #break

    #sys.exit(0)

    fresh = True

    for i, ins in enumerate(ins_list):
        print('instance id is ', ins.id)
        subfolder = os.path.join(folder, str(ins.id))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        else:
            if fresh:
                for root, dirs, files in os.walk(subfolder, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
        #break
        
        for k, submat_info in enumerate(ins.mat_set):
            m_id, xyxy, classname = submat_info
            #print('mid is ', m_id, ' xyxy is ', xyxy, ' classname is ', classname)
            mat = readmat(mat_path, mats, m_id)
            #print('mat shape is ', mat.shape)
            submat = mat[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]
            if save == 'npy':
                m_name = str(k) + '.npy'
                np.save(os.path.join(subfolder, m_name), submat)
            elif save == 'png':
                m_name = str(k) + '.png'
                cv2.imwrite(os.path.join(subfolder, m_name), submat)
            #print('submat shape is ', submat.shape)
            #break
            #print('submat is ', submat_info)
        if i > 2:
            break

