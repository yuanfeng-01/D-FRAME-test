"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
import pickle
from copy import deepcopy
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class NerVEDataset(DefaultDataset):
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
            with open(os.path.join(data_path, asset), 'rb') as f:
                temp_dict = pickle.load(f)
            # if not asset.endswith(".npy"):
            #     continue
            # if asset[:-4] not in self.VALID_ASSETS:
            #     continue
            # data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = temp_dict["pc"].astype(np.float32)
        # data_dict["color"] = data_dict["color"].astype(np.float32)
        # data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["segment"] = temp_dict["label"].astype(np.int32)

        # if "segment20" in data_dict.keys():
        #     data_dict["segment"] = (
        #         data_dict.pop("segment20").reshape([-1]).astype(np.int32)
        #     )
        # elif "segment200" in data_dict.keys():
        #     data_dict["segment"] = (
        #         data_dict.pop("segment200").reshape([-1]).astype(np.int32)
        #     )
        # else:
        #     data_dict["segment"] = (
        #         np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        #     )
        #
        # if "instance" in data_dict.keys():
        #     data_dict["instance"] = (
        #         data_dict.pop("instance").reshape([-1]).astype(np.int32)
        #     )
        # else:
        #     data_dict["instance"] = (
        #         np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        #     )
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict


@DATASETS.register_module()
class NerVEDatasetV2(DefaultDataset):
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
            with open(os.path.join(data_path, asset), 'rb') as f:
                temp_dict = pickle.load(f)

        data_dict["name"] = name
        data_dict["coord"] = temp_dict["pc"].astype(np.float32)
        data_dict["segment"] = temp_dict["direction"].astype(np.float32)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict


@DATASETS.register_module()
class NerVEDatasetV3(DefaultDataset):
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
            with open(os.path.join(data_path, asset), 'rb') as f:
                temp_dict = pickle.load(f)
            # if not asset.endswith(".npy"):
            #     continue
            # if asset[:-4] not in self.VALID_ASSETS:
            #     continue
            # data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = temp_dict["pc"].astype(np.float32)
        # data_dict["color"] = data_dict["color"].astype(np.float32)
        # data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["segment"] = temp_dict["label"].astype(np.int32)
        data_dict["direction"] = temp_dict["direction"].astype(np.float32)

        # if "segment20" in data_dict.keys():
        #     data_dict["segment"] = (
        #         data_dict.pop("segment20").reshape([-1]).astype(np.int32)
        #     )
        # elif "segment200" in data_dict.keys():
        #     data_dict["segment"] = (
        #         data_dict.pop("segment200").reshape([-1]).astype(np.int32)
        #     )
        # else:
        #     data_dict["segment"] = (
        #         np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        #     )
        #
        # if "instance" in data_dict.keys():
        #     data_dict["instance"] = (
        #         data_dict.pop("instance").reshape([-1]).astype(np.int32)
        #     )
        # else:
        #     data_dict["instance"] = (
        #         np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        #     )
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict


@DATASETS.register_module()
class NerVEDatasetV4(DefaultDataset):
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
            with open(os.path.join(data_path, asset), 'rb') as f:
                temp_dict = pickle.load(f)
            # if not asset.endswith(".npy"):
            #     continue
            # if asset[:-4] not in self.VALID_ASSETS:
            #     continue
            # data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = temp_dict["pc"].astype(np.float32)
        data_dict["direction"] = temp_dict["direction"].astype(np.float32)
        data_dict["segment"] = temp_dict["label"].astype(np.int32)

        data_dict["coord"] = data_dict["coord"][data_dict["segment"] == 1]
        data_dict["direction"] = data_dict["direction"][data_dict["segment"] == 1]
        data_dict["segment"] = data_dict["segment"][data_dict["segment"] == 1]

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict


@DATASETS.register_module()
class NerVEDatasetV5(DefaultDataset):
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
        pred = None
        for asset in assets:
            if asset.endswith(".pkl"):
                with open(os.path.join(data_path, asset), 'rb') as f:
                    temp_dict = pickle.load(f)
            else:
                pred = np.loadtxt(os.path.join(data_path, asset))
            # if not asset.endswith(".npy"):
            #     continue
            # if asset[:-4] not in self.VALID_ASSETS:
            #     continue
            # data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        ori_pc = temp_dict["pc"].astype(np.float32)
        if pred is None:
            cls_label = temp_dict["label"].astype(np.int32)
            data_dict["coord"] = ori_pc[cls_label == 1].astype(np.float32)
        else:
            cls_label = pred[:, -1].astype(np.int32)
            data_dict["coord"] = pred[:, :3][cls_label == 1].astype(np.float32)
        _, idx = self.get_knn_dis(data_dict["coord"], ori_pc, k=1)
        data_dict["direction"] = np.take(temp_dict["direction"].astype(np.float32), idx, axis=0)[:, 0, :]
        data_dict["segment"] = np.take(temp_dict["label"][:, None].astype(np.float32), idx, axis=0)[:, 0, 0]
        # data_dict["direction"] = temp_dict["direction"].astype(np.float32)
        # data_dict["segment"] = temp_dict["label"].astype(np.int32)

        # data_dict["coord"] = data_dict["coord"][data_dict["segment"] == 1]
        # data_dict["direction"] = data_dict["direction"][data_dict["segment"] == 1]
        # data_dict["segment"] = data_dict["segment"][data_dict["segment"] == 1]

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        name = data_dict.pop("name")
        data_dict = self.transform(data_dict)
        data_dict["name"] = name
        return data_dict

    def get_knn_dis(self, queries, pc, k):
        """
        queries [M, C]
        pc [P, C]
        """
        knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
        try:
            # print(pc)
            knn_search.fit(pc)
        except:
            knn_search.fit(pc.reshape(-1, 1))
        dis, knn_idx = knn_search.kneighbors(queries, return_distance=True)
        # res = np.take(pc, knn_idx, axis=0)  # M, K, C
        # res_dist = dis[knn_idx[0]]
        return dis, knn_idx



if __name__ == '__main__':
    dataset = NerVEDataset(data_root="/data2/lwj/PycharmProjects/MICCAI_TeethLand24/Pointcept/data/NerVE64Dataset")
    print(dataset[0]['name'], dataset[0]['coord'].shape, dataset[0]['segment'].shape)

# import numpy as np
# import pickle
# import os

# pred_path = './exp/nerve/semseg-pt-v3m1-0-base/result'
# gt_path = './data/nerve/test'
#
# prec = 0
# recall = 0
# num = 0
# for file_name in os.listdir(pred_path):
#     pred_file_path = os.path.join(pred_path, file_name)
#     if not os.path.isfile(pred_file_path):
#         continue
#     pred = np.loadtxt(pred_file_path)
#     # if len(pred) < 16384:
#     #     continue
#     num += 1
#     model_id = file_name.split('_')[0]
#     # gt_file_path = os.path.join(gt_path, model_id, 'cls_16384.pkl')
#     # with open(gt_file_path, 'rb') as f:
#     #     gt_data = pickle.load(f)
#     # gt_label = gt_data['label']
#     pred_label = pred[:, -1]
#     gt_label = pred[:, -2]
#     gt_edge_pts_sum = (gt_label == 1).sum()
#     pred_edge_pts_sum = (pred_label == 1).sum()
#     edge_pts_TP = np.logical_and(gt_label, pred_label).sum()
#     prec += (edge_pts_TP / pred_edge_pts_sum)
#     recall += (edge_pts_TP / gt_edge_pts_sum)
#
# print('Num: ', num)
# print('Precisioin: ', prec/num)
# print('Recall: ', recall/num)

# file_path = os.path.join(pred_path, '00000006_pred.txt')
# data = np.loadtxt(file_path)
# data = data[data[:,-1]==1]
# pts = data[:, :3]
# np.savetxt(os.path.join(pred_path, '0.txt'), pts)

# root_dir = '/data2/lwj/PycharmProjects/MICCAI_TeethLand24/Pointcept/data/nerve_v2'
# for cls_id in os.listdir(root_dir):
#     for model_name in os.listdir(os.path.join(root_dir, cls_id)):
#         file_name = os.path.join(root_dir, cls_id, model_name, 'cls_16384.pkl')
#         with open(file_name, 'rb') as f:
#             data = pickle.load(f)
#         direction = data['direction']
#         if (direction != 0).sum() == 0:
#             print(model_name)
