"Produce labels for existing mappoints with maskrcnn"

__author__ = 'Zherlock'

import os
import argparse
import time
import numpy as np
import cv2
import pickle
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from .instance import Instance
from .mappoint import MapPoint, Property


class LabelMps(object):
    """
    """

    def __init__(self, mp_path, mat_path, label_path):
        """Attributes: mp_path: A str for path of the mappoints txt
        mat_path: A str for path of the rgb frames
        label_path: A str for the path to store many temp results
        memo: A str for the path of the stored inference results
        """
        self.mp_path = mp_path
        self.mat_path = mat_path
        self.label_path = label_path

    def init_mps(self):
        """read mappoints in the files
        return a list of map points
        """
        if not os.path.exists(self.mp_path):
            print('no slam mps results, return.')
            return 1
        cnt = 0
        self.points = []
        self.keyframe_set = set()
        last_mp = -1
        with open(self.mp_path, "r") as f:
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
                    self.points.append(p)
                    p.info[m_id] = Property()
                    p.info[m_id].x = x_cor
                    p.info[m_id].y = y_cor
                    cnt += 1
                else:
                    p.info[m_id] = Property()
                    p.info[m_id].x = x_cor
                    p.info[m_id].y = y_cor
                self.keyframe_set.add(m_id)
        print("{} key frames and {} mppoints are envolved".format(
            len(self.keyframe_set), cnt))

    def init_mats(self):
        """read mat file names in the folder
        save them in a list
        """
        dirs = os.listdir(self.mat_path)
        self.mats = sorted(dirs, key=lambda x: float(x[:-4]))  # 所有图片文件名
        assert self.mats[0] == '00001.png'

    def create_text_labels(classes, class_names):
        """
        Args:
            classes (list[int] or None):
            class_names (list[str] or None):

        Returns:
            list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        return labels

    def init_predictor(self):
        """init a mask rcnn predictor using detectron2
        """
        cfg = get_cfg()
        # load values from a file
        cfg.merge_from_file(
            "/home/zherlock/InstanceDetection/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.merge_from_list(
            ["MODEL.WEIGHTS", "/home/zherlock/InstanceDetection/detectron2/pre_train_model/model_final_f10217.pkl", "MODEL.DEVICE", "cpu"])
        # print(cfg.dump())
        self.predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.classnames = metadata.get("thing_classes", None)  # 所有label名,80类物体哇

    def init_memo(self):
        """fetch inference result of all mats, if memo exists, read from memo, 
        if not, use the model to inference and save it to memo
        """
        try:
            self.memo = LabelMps.load_obj(os.path.join(self.label_path, 'memo'))
            print('we get memo')
        except:
            print('no memo, u need to compute it')
            self.memo = {}
            t0 = time.time()
            for m_id in range(len(self.mats)):
                # pass
                if m_id not in self.keyframe_set:
                    continue
                im = self.readmat(m_id)
                op = self.inference(m_id, im)
                self.memo[m_id] = op
                if m_id > 10:
                    pass
            LabelMps.save_obj(self.memo, os.path.join(self.label_path, 'memo'))
            print('it takes ', time.time() - t0, " secs for memo computing, till now is good.")

    def inference(self, m_id: int, im=None):
        """fetch inference result of a mat, if memo exists, read from memo, 
        if not, use the model to inference and save it to memo
        """
        if m_id in self.memo:
            return self.memo[m_id]
        output = self.predictor(im)
        self.memo[m_id] = output
        return output

    def readmat(self, m_id: int):
        """fetch the mat using cv2.imread
        """
        return cv2.imread(os.path.join(self.mat_path + self.mats[m_id]))

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

    def filter_p(self, points: list):
        """filter map points that have never been recognized in an object 
        return a list of map points that have been recognized in an object
        """
        thre = 0.6
        res = []
        for p in points:
            for m_id in p.info.keys():
                op = self.inference(m_id, None)
                for i in range(len(op["instances"])):
                    if op["instances"].scores[i] > thre:
                        if op["instances"].pred_masks[i][p.info[m_id].y][p.info[m_id].x]:
                            res.append(p)
                            break
                else:
                    continue
                break
        return res

    def fresh(points: list):
        """make all points fresh
        """
        for p in points:
            p.fresh = True
            p.instance_id = -1

    def convisual(pa: MapPoint, pb: MapPoint):
        """return the convisual mats that pa and pb both appeared
        return a set of mids
        """
        set_a = set([m_id for m_id in pa.info.keys()])
        set_b = set([m_id for m_id in pb.info.keys()])
        return set_a & set_b

    def same_instance(self, p: MapPoint, q: MapPoint):
        """judge if two map points belong to the same instance
        if true, return Ture and mat_set 
        """
        # same instance的同时，保存子图的信息，用作之后建立每个instance的图形库建立
        thre = 0.5
        acc = 0
        conv = LabelMps.convisual(p, q)
        min_req = 3
        mat_set = set()
        if not conv or len(conv) < min_req:
            #print('no conv')
            return False, mat_set
        for m_id in conv:
            op = self.inference(m_id, None)
            for i in range(len(op["instances"])):
                if op["instances"].scores[i] > thre:
                    if op["instances"].pred_masks[i][p.info[m_id].y][p.info[m_id].x]:
                        if op["instances"].pred_masks[i][q.info[m_id].y][q.info[m_id].x]:
                            if 'lables' not in op:
                                op['labels'] = LabelMps.create_text_labels(
                                    op["instances"].pred_classes, self.classnames)
                                if op['labels'][i] is not 'person':
                                    temp = op['instances'].pred_boxes[i].tensor[0]
                                    xyxy = tuple([int(temp[k]) for k in range(4)])
                                    mat_set.add(
                                        tuple((m_id, xyxy, op['labels'][i])))
                                    acc += 1
        if acc >= min_req:
            return True, mat_set
        else:
            return False, mat_set

    def merge(self, p: MapPoint, q: MapPoint, mat_set: set):
        """give two map points the same instance_id
        """
        if p.fresh and not q.fresh:
            p.instance_id = q.instance_id
            self.instance_list[p.instance_id].mps.add(p.global_id)
        elif q.fresh and not p.fresh:
            q.instance_id = p.instance_id
            self.instance_list[q.instance_id].mps.add(q.global_id)
        else:
            ins = Instance()
            self.instance_list.append(ins)
            ins.id = len(self.instance_list)-1
            p.instance_id = ins.id
            q.instance_id = ins.id
            ins.mps.add(p.global_id)
            ins.mps.add(q.global_id)
        p.fresh = False
        q.fresh = False
        self.instance_list[p.instance_id].mat_set = self.instance_list[p.instance_id].mat_set.union(
            mat_set)

    def statics(self):
        print('cluster {} instances'.format(len(self.instance_list)))
        print('Their sizes are ')
        cnt = 0
        acc = 0
        for ins in self.instance_list:
            print(cnt, '   ', len(ins.mps), '  ', ins.id)
            acc += len(ins.mps)
            cnt += 1
        print('sum points is', acc)

    def instance_labeling(self):
        # 现在看这个函数觉得很奇怪，point[下标为列表下标], ins.mps存储的是point的global_id，这两个应该联系不起来的
        # 的确是联系起来的，point是一个个加入列表的，列表下标刚好和point的global_id一样
        for ins in self.instance_list:
            for p_id in ins.mps:
                self.points[p_id].instance_id = ins.id

    def labeling(self):
        self.init_mps()
        self.init_mats()
        self.init_predictor()
        self.init_memo()
        pps = self.filter_p(self.points)
        LabelMps.fresh(pps)
        self.ins_path = os.path.join(self.label_path, 'InsList.npy')
        try:
            self.instance_list = np.load(self.ins_path).tolist()
            print('lucky, u have existed results')
        except Exception:
            print('u need to merge instances urself')
            self.instance_list = []
            start = time.time()
            for p in pps:
                for q in pps:
                    if p.fresh or q.fresh:
                        if p.global_id != q.global_id:
                            a, mat_set = self.same_instance(p, q)
                            if a:
                                self.merge(p, q, mat_set)

            end = time.time()
            print("it takes {} seconds to merge instances".format(end - start))
            temp = np.array(self.instance_list)
            np.save(self.ins_path, temp)
        self.statics()
        self.instance_labeling()
        cnt = 0
        # for p in points:
        #    print(p.instance_id)
        with open(os.path.join(self.label_path, "instance_saving.txt"), "w") as f:
            for p in self.points:
                if p.instance_id != -1:
                    f.write(str(p.global_id))
                    f.write(' ')
                    f.write(str(p.instance_id))
                    f.write('\n')
                    cnt += 1
        print('total, ', cnt, ' labeled points')
        print('program exit with 0')


    

    


