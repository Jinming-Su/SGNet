# SGNet
Project for the IJCAI 2021 paper "Structure Guided Lane Detection"

## Abstract
Recently, lane detection has made great progress
with the rapid development of deep neural networks
and autonomous driving. However, there
exist three mainly problems including characterizing
lanes, modeling the structural relationship between
scenes and lanes, and supporting more attributes
(e.g., instance and type) of lanes. In this
paper, we propose a novel structure guided framework
to solve these problems simultaneously. In
the framework, we first introduce a new lane representation
to characterize each instance. Then a topdown
vanishing point guided anchoring mechanism
is proposed to produce intensive anchors, which efficiently
capture various lanes. Next, multi-level
structural constraints are used to improve the perception
of lanes. In the process, pixel-level perception
with binary segmentation is introduced to
promote features around anchors and restore lane
details from bottom up, a lane-level relation is put
forward to model structures (i.e., parallel) around
lanes, and an image-level attention is used to adaptively
attend different regions of the image from the
perspective of scenes. With the help of structural
guidance, anchors are effectively classified and regressed
to obtain precise locations and shapes. Extensive
experiments on public benchmark datasets
show that the proposed approach outperforms stateof-
the-art methods with 117 FPS on a single GPU.

## Method
![Framework](https://github.com/Jinming-Su/SGNet/blob/master/asserts/framework.png)
Framework of our approach. We first extract the common features by the extractor, which provides features for vanishing point
guided anchoring and pixel-level perception. The anchoring produces intensive anchors and perception utilizes binary segmentation to
promote features around lanes. Promoted features are used to classify and regress anchors with the aid of lane-level relation and image-level
attention. The dashed arrow indicates the supervision, and the supervision of vanishing point and lane segmentation is omitted in the figure.

## Quantitative Evaluation
![Quantitative Evaluation](https://github.com/Jinming-Su/SGNet/blob/master/asserts/performance.png)

## Qualitative Evaluation
![Qualitative Evaluation](https://github.com/Jinming-Su/SGNet/blob/master/asserts/vialization.png)

## Usage
### Dataset Convertion
For CULane, run 
```
python datasets/2_generate_vp_label_dist_culane.py
```

For Tusimple, run 
```
.datasets/gen_tusimple.sh
```

### NMS Installation
```
cd lib/nms; python setup.py install
```

### Training
```
python main.py train --exp_name workdir --cfg cfgs/resnet34.py
```

### Testing
```
python main.py test --exp_name workdir --cfg cfgs/resnet34.py
```

### Evaluation
```
cd evaluateion/lane_evaluation
make
./run.sh 
./run_all.sh
```

### Visualization
```
python main.py test -exp_name workdir --view all
```
Thanks for the  reference provided by the [smart code](https://github.com/lucastabelini/LaneATT).

## Citation
```
@inproceedings{su2021structure,
  title={Structure Guided Lane Detection},
  author={Su, Jinming and Chen, Chao and Zhang, Ke and Luo, Junfeng and Wei, Xiaoming and Wei, Xiaolin},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2021}
}
```
