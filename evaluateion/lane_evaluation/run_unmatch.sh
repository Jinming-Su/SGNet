data_dir=../../dataset/CULane/
detect_dir=/workdir/chenchao/data/CULane/anno_new/
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list=test_unmatch.txt
out=out.txt
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out
