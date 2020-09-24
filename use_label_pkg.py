"""test label pkg
label功能似乎没问题了。
"""

__author__ = 'zherlock'

import label_pkg
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=os.getcwd(), help='parent directionary of label folder')
    parser.add_argument('--func', type=int, default=0, help='0 for merge instances and 1 for showing instance and 2 for generate final label')
    opt = parser.parse_args()
    label_path = os.path.join(opt.path, 'label/')
    mp_path = os.path.join(label_path, 'scmantic_saving.txt')#file path of slam result
    mat_path = os.path.join(opt.path, 'rgb/')
    if opt.func == 0:
        lms = label_pkg.LabelMps(mp_path, mat_path, label_path)
        sys.exit(lms.labeling())
    elif opt.func == 1:
        pass
        #sys.exit(show(opt))
    else:
        pass
        #sys.exit(merge_label(opt))
    
    try:
        pass
        #Instance_list = np.load(Ins_path).tolist()
        #print('lucky, u have existed results')
    except Exception:
        print('fail to read InsList.py')