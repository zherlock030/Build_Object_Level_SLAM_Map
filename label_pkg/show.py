"""display the map point of each instance
"""

__author__ = 'zherlock'

from .utils import *
import os

class ShowInstanceMps(object):
    def __init__(self, mp_path, mat_path, label_path):
        self.mp_path = mp_path
        self.mat_path = mat_path
        self.label_path = label_path
        self.ins_path = os.path.join(self.label_path, 'InsList.npy')
        self.points, _ = init_mps(self.mp_path)
        self.mats = init_mats(self.mat_path)
        try:
            self.ins_list = np.load(self.ins_path).tolist()
        except Exception:
            print("no instance list could be loaded")

    def visual_instance(self, k:int, points:list):
        """input: k->instance id, points: list of all map points
        """
        cnt = 0
        ins = self.ins_list[k]
        for p_id in ins.mps:
            if cnt % 1 == 0:
                p = points[p_id]
                visualize_p(p, self.mats, self.mat_path)
            cnt += 1
    def showing(self, k:int):
        """show the map point of instance k
        """
        self.visual_instance(k, self.points)
        
