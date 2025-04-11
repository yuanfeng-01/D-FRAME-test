import torch.nn as nn
import torch.nn.functional
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)  # [BN, 3]
        # seg_logits = torch.nn.functional.adaptive_max_pool1d(seg_logits.permute(0, 2, 1), M).permute(0, 2, 1)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.pred_head = (
            nn.Linear(backbone_out_channels, 3)
        )
        self.backbone = build_model(backbone)
        assert len(criteria) > 1
        self.criteria = build_criteria(criteria[0])
        self.criteria_pred = build_criteria(criteria[1])

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        pred_res = self.pred_head(feat)
        # train
        if self.training:
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_pred(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred)
        # eval
        elif "segment" in input_dict.keys():
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_pred(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred, seg_logits=seg_logits, pred=pred_res)
        # test
        else:
            return dict(seg_logits=seg_logits, pred=pred_res)


@MODELS.register_module()
class DefaultSegmentorV4(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.pred_head = (
            nn.Linear(backbone_out_channels, 3)
        )
        self.backbone_cls = build_model(backbone)
        self.backbone_field = build_model(backbone)
        assert len(criteria) == 2
        self.criteria_cls = build_criteria(criteria[0])
        self.criteria_field = build_criteria(criteria[1])

    def forward(self, input_dict):
        point = Point(input_dict)
        point_cls = self.backbone_cls(point)
        point_field = self.backbone_field(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point_cls, Point) and isinstance(point_field, Point):
            feat_cls = point_cls.feat
            feat_field = point_field.feat
        else:
            feat_cls = point_cls.feat
            feat_field = point_field.feat
        seg_logits = self.seg_head(feat_cls)
        pred_res = self.pred_head(feat_field)
        # train
        if self.training:
            loss_seg = self.criteria_cls(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_field(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred)
        # eval
        elif "segment" in input_dict.keys():
            loss_seg = self.criteria_cls(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_field(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred, seg_logits=seg_logits, pred=pred_res)
        # test
        else:
            return dict(seg_logits=seg_logits, pred=pred_res)
        
@MODELS.register_module()
class DefaultSegmentorV5(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.pred_head = (
            nn.Linear(backbone_out_channels, 3)
        )
        self.backbone = build_model(backbone)
        assert len(criteria) > 1
        self.criteria = build_criteria(criteria[0])
        self.criteria_pred = build_criteria(criteria[1])
        self.criteria_smooth = build_criteria(criteria[2])

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        pred_res = self.pred_head(feat)
        # train
        if self.training:
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_pred(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss_smooth = self.criteria_smooth([pred_res, input_dict["coord"], input_dict["sampled_num"]], [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred + loss_smooth
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred, loss_smooth=loss_smooth)
        # eval
        elif "segment" in input_dict.keys():
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_pred = self.criteria_pred(pred_res, [input_dict["direction"], input_dict["segment"]])
            loss_smooth = self.criteria_smooth([pred_res, input_dict["coord"], input_dict["sampled_num"]], [input_dict["direction"], input_dict["segment"]])
            loss = loss_seg + loss_pred + loss_smooth
            return dict(loss=loss, loss_seg=loss_seg, loss_pred=loss_pred, loss_smooth=loss_smooth, seg_logits=seg_logits, pred=pred_res)
        # test
        else:
            return dict(seg_logits=seg_logits, pred=pred_res)
        
@MODELS.register_module()
class DefaultSegmentorV6(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.pred_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        assert len(criteria) >= 1
        self.criteria = build_criteria(criteria[0])

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        pred_res = self.pred_head(feat)  # [BN, 3]
        # seg_logits = torch.nn.functional.adaptive_max_pool1d(seg_logits.permute(0, 2, 1), M).permute(0, 2, 1)
        # train
        if self.training:
            loss = self.criteria(pred_res, input_dict["direction"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(pred_res, input_dict["direction"])
            return dict(loss=loss, pred=pred_res)
        # test
        else:
            return dict(pred=pred_res)
        
@MODELS.register_module()
class DefaultSegmentorV7(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.pred_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        assert len(criteria) == 2
        self.criteria = build_criteria(criteria[0])
        self.criteria_smooth = build_criteria(criteria[1])

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        pred_res = self.pred_head(feat)  # [BN, 3]
        # seg_logits = torch.nn.functional.adaptive_max_pool1d(seg_logits.permute(0, 2, 1), M).permute(0, 2, 1)
        # train
        if self.training:
            loss_pred = self.criteria(pred_res, input_dict["direction"])
            loss_smooth = self.criteria_smooth([pred_res, input_dict["coord"], input_dict["sampled_num"]], input_dict["direction"])
            loss = loss_pred + loss_smooth
            return dict(loss=loss, loss_pred=loss_pred, loss_smooth=loss_smooth)
        # eval
        elif "segment" in input_dict.keys():
            loss_pred = self.criteria(pred_res, input_dict["direction"])
            loss_smooth = self.criteria_smooth([pred_res, input_dict["coord"], input_dict["sampled_num"]], input_dict["direction"])
            loss = loss_pred + loss_smooth
            return dict(loss=loss, loss_pred=loss_pred, loss_smooth=loss_smooth, pred=pred_res)
        # test
        else:
            return dict(pred=pred_res)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
