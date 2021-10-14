# Model settings
val_every: 1000
model_checkpoint_interval: 1
seed: 0
model:
  name: VP_lane_attenadd_fpn_more_conv_refine_laneseg_lane_level_linear_matrix_gaussian
  parameters:
    backbone: resnet34
    S: &S 72
    topk_anchors: 1000
    anchors_freq_path: 'data/culane_anchors_freq.pt'
    img_h: &img_h 360
    img_w: &img_w 640
    w_interval: 5.0
batch_size: 8
epochs: 20
loss_parameters: {}
train_parameters:
  conf_threshold:
  nms_thres: 15.
  nms_topk: 3000
test_parameters:
  conf_threshold: 0.5
  nms_thres: 50.
  nms_topk: &max_lanes 4
optimizer:
  name: Adam
  parameters:
    lr: 0.0003
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: 222200

# Dataset settings
datasets:
  train:
    type: LaneDataset
    parameters:
      S: *S
      dataset: culane
      split: train
      img_size: [*img_h, *img_w]
      max_lanes: *max_lanes
      normalize: false
      aug_chance: 1.0
      augmentations:
         - name: Affine
           parameters:
             translate_px:
              x: !!python/tuple [-25, 25]
              y: !!python/tuple [-10, 10]
             rotate: !!python/tuple [-6, 6]
             scale: !!python/tuple [0.85, 1.15]
         - name: HorizontalFlip
           parameters:
             p: 0.5

      root: "datasets/culane"
  
  train_1000:
    type: LaneDataset
    parameters:
      S: *S
      dataset: culane
      split: train
      img_size: [*img_h, *img_w]
      max_lanes: *max_lanes
      normalize: false
      aug_chance: 1.0
      augmentations:
         - name: Affine
           parameters:
             translate_px:
              x: !!python/tuple [-25, 25]
              y: !!python/tuple [-10, 10]
             rotate: !!python/tuple [-6, 6]
             scale: !!python/tuple [0.85, 1.15]
         - name: HorizontalFlip
           parameters:
             p: 0.5

      root: "datasets/culane"
  test:
    type: LaneDataset
    parameters:
      S: *S
      dataset: culane
      split: test
      img_size: [*img_h, *img_w]
      max_lanes: *max_lanes
      normalize: false
      aug_chance: 0
      augmentations:
      root: "datasets/culane"
  
  test_1000:
    type: LaneDataset
    parameters:
      S: *S
      dataset: culane
      split: test
      img_size: [*img_h, *img_w]
      max_lanes: *max_lanes
      normalize: false
      aug_chance: 0
      augmentations:
      root: "datasets/culane"  

  val:
    type: LaneDataset
    parameters:
      S: *S
      dataset: culane
      split: val
      img_size: [*img_h, *img_w]
      max_lanes: *max_lanes
      normalize: false
      aug_chance: 0
      augmentations:
      root: "datasets/culane"
