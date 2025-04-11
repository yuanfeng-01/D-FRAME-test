import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc,m,centroid

@DATASETS.register_module()
class TeethLandDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "normal",
	"curvature",
        "segment",
    ]
    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            # with open(os.path.join(data_path, asset), 'rb') as f:
            #     temp_dict = pickle.load(f)
            # if not asset.endswith(".npy"):hgh
            #     continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.loadtxt(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        # data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["curvature"] = data_dict["curvature"][:, None].astype(np.float32)
        data_dict["segment"] = data_dict["segment"].astype(np.float32).T
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict


if __name__ == '__main__':
    # dataset = TeethLandDataset(data_root="/data2/lwj/PycharmProjects/MICCAI_TeethLand24/Pointcept/data/teeth_land")
    # print(dataset[0]['name'], dataset[0]['coord'].shape, dataset[0]['normal'].shape, dataset[0]['segment'].shape)
    data_root = '/data2/lwj/PycharmProjects/new/test'
    save_root = '/data2/lwj/PycharmProjects/MICCAI_TeethLand24/Pointcept/data/teeth_land'
    # data_lst = []
    # for cls in os.listdir(data_root):
    #     cls_path = os.path.join(data_root, cls)
    #     for model_id in os.listdir(cls_path):
    #         model_path = os.path.join(cls_path, model_id)
    #         for file_name in os.listdir(model_path):
    #             data_lst.append(os.path.join(model_path, file_name))
    # random.shuffle(data_lst)
    # num_files = len(data_lst)
    # print(num_files)
    # train_data_lst = data_lst[:int(num_files*0.8)]
    # val_data_lst = data_lst[int(num_files*0.8):int(num_files*0.9)]
    # test_data_lst = data_lst[int(num_files*0.9):]
    # for cls in os.listdir(data_root):
    #     cls_path = os.path.join(data_root, cls)
    #     for model_id in os.listdir(cls_path):
    #         model_path = os.path.join(cls_path, model_id)
    #         for file_name in os.listdir(model_path):
    #             file_path = os.path.join(model_path, file_name)
    #             pc_path = os.path.join(model_path, file_name)
    #             # conf_path = pc_path.replace('pc_normal', 'confidence')
    #             pc = np.loadtxt(pc_path)[:, :3]
    #             normal = np.loadtxt(pc_path)[:, 3:6]
    #             # conf = np.loadtxt(conf_path).T
    #             conf = np.zeros_like(np.concatenate([pc, normal], axis=-1))
    #             pc, _, _ = pc_normalize(pc)
    #             normal /= (np.sqrt(np.sum(normal ** 2, axis=-1))[:, None])
    #             save_path = os.path.join(save_root, 'test', model_id + f'_{cls}_{file_name[:-4]}')
    #             # if file_path in train_data_lst:
    #             #     save_path = os.path.join(save_root, 'train', model_id+f'_{cls}_{file_name[:-4]}')
    #             # elif file_path in val_data_lst:
    #             #     save_path = os.path.join(save_root, 'val', model_id+f'_{cls}_{file_name[:-4]}')
    #             # else:
    #             #     save_path = os.path.join(save_root, 'test', model_id + f'_{cls}_{file_name[:-4]}')
    #             save_pc_path = os.path.join(save_path, 'coord.txt')
    #             save_conf_path = save_pc_path.replace('coord', 'segment')
    #             save_normal_path = save_pc_path.replace('coord', 'normal')
    #             os.makedirs(save_path, exist_ok=True)
    #             np.savetxt(save_pc_path, pc)
    #             np.savetxt(save_normal_path, normal)
    #             np.savetxt(save_conf_path, conf)