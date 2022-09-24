# model settings
'''SwinTiny
1)backbone:
embed_dim = 96
depths = [2, 2, 6, 2]
num_heads = [3, 6, 12, 24]
2)fpn
in_channel = [96, 192, 384, 768]
'''

'''SwinSmall
1)backbone:
depths = [2, 2, 18, 2]
'''

pretrained = '/home/fzh/Templates/JDET/swin_tiny_patch4_window7_224_22k_cvt.pth'  # noqa
model = dict(
    type='S2ANet',
    backbone=dict(
        type='SwinTiny',  # SwinSmall
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint = False,
        # with_cp=False,
        # convert_weights=True,
       pretrained=True,
       pth_file=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_thr=0.1),
            max_per_img=2000),
        train_cfg=dict(
            fam_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                target_means=(0., 0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1., 1.),
                                clip_border=True),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            odm_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                target_means=(0., 0., 0., 0., 0.),
                                target_stds=(1., 1., 1., 1., 1.),
                                clip_border=True),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        )
    )

dataset = dict(
    train=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir='/data0/fzh/uncommon/JDet-dataset/preprocessed/train_1024_200_1.0',
        transforms=[
            dict(
                type="AugmentHSV",  # reference yolo
                hgain=0.015,
                sgain=0.7,
                vgain=0.4
            ),
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type='RotatedRandomFlip',
                direction="horizontal",
                prob=0.5),
            dict(
                type='RotatedRandomFlip',
                direction="vertical",
                prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, )

        ],
        batch_size=1,
        num_workers=2,
        shuffle=True,
        filter_empty_gt=False,
        balance_category=True
    ),
    val=dict(
        type="FAIR1M_1_5_Dataset",
        dataset_dir='/data0/fzh/uncommon/JDet-dataset/preprocessed/train_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, ),
        ],
        batch_size=1,
        num_workers=2,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir='/data0/fzh/uncommon/JDet-dataset/preprocessed/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, ),
        ],
        dataset_type="FAIR1M_1_5",
        num_workers=4,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD',
    lr=0.01, #0.01/2., #0.0,#0.01*(1/8.),
    momentum=0.9,
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35,
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[7, 10])


logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 12
eval_interval = 12
checkpoint_interval = 1
log_interval = 50

