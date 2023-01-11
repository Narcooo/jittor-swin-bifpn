# {DATASET_PATH}
# model settings
pretrained_weights = '/work_dirs/faster_rcnn_RoITrans_swint_fpn_fair_15' \
                     '/checkpoints/ckpt_36.pkl',
# model settings
model = dict(
    type='S2ANet',
    backbone=dict(
        type='swin_tiny_patch4_window7_224',
        # frozen_stages=1,
        dilation=False,
        pretrained=False),
    neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes=11,
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
dataset_type = 'FAIR1M_1_5_Dataset'
overlap = 512
ms = '1.0'
test_ms = '0.5-1.0-1.5'
size = 1024
train = 'train'
dataset_root = f'/data/preprocessed_FAIR1m15_512'
dataset = dict(
    train=dict(
        type=dataset_type,
        dataset_dir=f'{dataset_root}/{train}_{size}_{overlap}_{ms}',
        transforms=[
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
            # dict(
            #     type="RandomRotateAug",
            #     random_rotate_on=True,
            # ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False, )

        ],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=True,
        balance_category=True
    ),
    val=dict(
        type=dataset_type,
        dataset_dir=f'{dataset_root}/val_{size}_{overlap}_{ms}',
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
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        dataset_type="FAIR1M_1_5",
        images_dir=f'{dataset_root}/test_{size}_{overlap}_{test_ms}/images',
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
        num_workers=4,
        batch_size=1,
    )
)


optimizer = dict(type='AdamW',
                 lr=0.0001,
                 betas=(0.9, 0.999),
                 weight_decay=0.05,
                 )

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    milestones=[23, 31])

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 36
# max_iter = 30
eval_interval = 100
checkpoint_interval = 1
log_interval = 100

# when we the trained model from cshuan, image is rgb

