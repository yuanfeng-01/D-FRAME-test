weight = 'exp/nerve_v5/semseg-pt-v3m1-0-base/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 50380398
save_path = 'exp/nerve_v5/semseg-pt-v3m1-0-base'
num_worker = 24
batch_size = 8
batch_size_val = None
batch_size_test = None
epoch = 800
eval_epoch = 800
sync_bn = False
enable_amp = True
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=5e-05)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='EdgeEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='EdgeTesterV3', verbose=True)
model = dict(
    type='DefaultSegmentorV6',
    num_classes=3,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D')),
    criteria=[[{
        'type': 'CosineLoss',
        'reduction': 'none',
        'loss_weight': 1.0
    }]])
optimizer = dict(type='AdamW', lr=0.0005, weight_decay=0.05)
scheduler = dict(
    type='MultiStepLR', milestones=[0.05, 0.2, 0.4, 0.6, 0.8], gamma=0.5)
dataset_type = 'NerVEDatasetV5'
data_root = 'data/nerve_v7_noisy_0.005'
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=['x', 'y', 'z'],
    train=dict(
        type='NerVEDatasetV5',
        split='train',
        data_root='data/nerve_v7_noisy_0.005',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment', 'direction'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'direction'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='NerVEDatasetV5',
        split='val',
        data_root='data/nerve_v7_noisy_0.005',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment', 'direction'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'direction'),
                feat_keys=('coord', ))
        ],
        test_mode=False),
    test=dict(
        type='NerVEDatasetV5',
        split='test_nerve',
        data_root='data/nerve_v7_noisy_0.005',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment', 'direction'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'direction'),
                feat_keys=('coord', ))
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'segment', 'direction'),
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', ))
            ],
            aug_transform=[])))
