data_dir=../../dataset/CULane/
data_dir=../../dataset/CULane/
#detect_dir=/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/LaneATT-main_offset_pred_vp/experiments/vp_angle_5_aug_pred_vp_34_lane_attenadd_fpn_20e_more_conv_refien_laneseg/results/epoch_0010_49/test_predictions/
detect_dir=/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/LaneATT-main_offset_pred_vp/experiments/vp_angle_5_aug_pred_vp_18_lane_attenadd_20e_refine_seg/results/epoch_0012/test_predictions/
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list=../../dataset/CULane/list/test.txt
out=out.txt
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out
