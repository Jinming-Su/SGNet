data_dir=../../dataset/CULane/
detect_dir=/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/LaneATT-main_offset_pred_vp/experiments/vp_angle_5_aug_pred_vp_18_lane_attenadd_20e_refine_seg/results/epoch_0012/test_predictions/
#detect_dir=/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/LaneATT-main_offset_pred_vp/experiments/vp_angle_5_aug_pred_vp_18_lane_attenadd_20e_refine_seg/results/epoch_0014/test_predictions/
#detect_dir=/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/LaneATT-main_offset_pred_vp/experiments/vp_angle_5_aug_pred_vp_34_lane_attenadd_fpn_20e_more_conv_refien_laneseg/results/epoch_0010_49/test_predictions/
list_dir=../../dataset/CULane/list/

output='./'
exp=density
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list0=${list_dir}/test_split/test0_normal.txt
list1=${list_dir}/test_split/test1_crowd.txt
list2=${list_dir}/test_split/test2_hlight.txt
list3=${list_dir}/test_split/test3_shadow.txt
list4=${list_dir}/test_split/test4_noline.txt
list5=${list_dir}/test_split/test5_arrow.txt
list6=${list_dir}/test_split/test6_curve.txt
list7=${list_dir}/test_split/test7_cross.txt
list8=${list_dir}/test_split/test8_night.txt
out0=${output}/out0_normal.txt
out1=${output}/out1_crowd.txt
out2=${output}/out2_hlight.txt
out3=${output}/out3_shadow.txt
out4=${output}/out4_noline.txt
out5=${output}/out5_arrow.txt
out6=${output}/out6_curve.txt
out7=${output}/out7_cross.txt
out8=${output}/out8_night.txt
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out0
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out1
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out2
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out3
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out4
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out5
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out6
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out7
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out8
cat ${output}/out*.txt>./${output}/${exp}_iou${iou}_split.txt
