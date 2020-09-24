"""remerge instances in instance_list.
"""

__author__ = 'zherlock'

from .utils import *
import os
import numpy as np

class ReMerge(object):
    def __init__(self, mp_path, label_path):
        self.label_path = label_path
        self.mp_path = mp_path
        self.ins_path = os.path.join(label_path, 'InsList.npy')
        try:
            self.ins_list = np.load(self.ins_path).tolist()
            print('instance list loaded')
        except Exception:
            print('fail to load instance list')
        self.points, _ = init_mps(self.mp_path)
    
    def remerge(self):
        """give instances new ids, and update map points' instance id
        """
        self.human_label_path = os.path.join(self.label_path, 'human label.txt')
        if not os.path.exists(self.human_label_path):
            print('no human label founded, remerge failed.')
            return 1
        name2id = {}  # classname: [id], read from human label
        id2name = {}  # id: classname, read from human label  
        id2id = {}  # old id: new id, new id is from 0 to N.
        with open(self.human_label_path, 'r') as f:
            for line in f:
                temp = line.split(':') # temp[0] is instance id, temp[1] is instance name
                temp[0] = int(temp[0])
                id2name[temp[0]] = temp[1]
                if temp[1] in name2id:
                    name2id[temp[1]].append(temp[0])
                else:
                    name2id[temp[1]] = [temp[0]]
        self.match_path = os.path.join(self.label_path, 'match_table_new.txt')
        with open(self.match_path, 'w') as f:
            for i, (name, ids) in enumerate(name2id.items()):
                f.write(str(i))
                f.write(':')
                f.write(name)
                f.write('\r\n')  # write matching table for SLAM, instance id: classname 
                for id in ids:
                    id2id[id] = i # old_id: new_id
        for ins in self.ins_list:
            if ins.id in id2id:
                ins.id = id2id[ins.id]  # to new id
            else:
                ins.id = -1  # instance not in human label because of too few points or background
        instance_labeling(self.ins_list, self.points)
        save_results(self.points, self.label_path)

        





