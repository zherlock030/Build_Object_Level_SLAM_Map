"""
For each submat of each instance, we extract the corresponding deep features, and for the 
whole instance, we store these features in one txt file.
"""

__author__ = 'zherlock'

from mat_feature_storage import *

def read_feature(fpath):
    return np.load(fpath)

if __name__ == '__main__':
    ins_path = '/home/zherlock/SLAM/build_my_dataset/zt301/0/label/InsList.npy'
    ins_list = np.load(ins_path).tolist()
    feature_folder = '/home/zherlock/SLAM/build_my_dataset/feature_memo/'
    subfeature_folder = '/home/zherlock/SLAM/build_my_dataset/subfeatures'
    if not os.path.exists(subfeature_folder):
        os.makedirs(subfeature_folder)
    for ins in ins_list:
        ins.mat_set = clean(ins.mat_set)
        fpath = os.path.join(subfeature_folder, str(ins.id) + '.txt')
        with open(fpath, 'w') as f:
            for submat_info in ins.mat_set:
                m_id, xyxy, classname = submat_info
                folder = os.path.join(feature_folder, str(m_id) + '.npy')
                feature = read_feature(folder)
                #print('feature ', feature.shape)
                ratio = 480 // feature.shape[2]
                xyxy = [t // ratio for t in xyxy]
                #print('xyxy is ', xyxy)
                subfeature = feature[:, :, xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                #print('subfeature ', subfeature.shape)
                subfeature = torch.from_numpy(subfeature)
                pool_feature = torch.nn.functional.adaptive_max_pool2d(input = subfeature, output_size = 1)
                pool_feature = torch.squeeze(pool_feature)
                #print('maxpool,  ', pool_feature.shape)
                for i in range(pool_feature.shape[0] - 1):
                    f.write(str(pool_feature[i].item()))
                    f.write(',')
                f.write(str(pool_feature[-1].item()))
                f.write('\r\n')
        




