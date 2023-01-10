
# Online Relational Tracking With Camera Motion Suppression

## Introduction

This repository contains code for *Online Relational Tracking With Camera Motion Suppression* (FOR_TRACKING). The paper is accepted in the "Journal of Visual Communication and Image Representation" and can be accessed from [this link](https://www.sciencedirect.com/science/article/pii/S104732032200270X).

To overcome challenges in multiple object tracking tasks, recent algorithms use interaction cues alongside motion and appearance features. These algorithms use graph neural networks or transformers to extract interaction features that lead to high computation costs. In this paper, a novel interaction cue based on geometric features is presented aiming to detect occlusion and re-identify lost targets with low computational cost. 

Moreover, in most algorithms, camera motion is considered negligible, which is a strong assumption that is not always true and leads to ID Switches or mismatching of targets. In this paper, a method for measuring camera motion and removing its effect is presented that efficiently reduces the camera motion effect on tracking.

The proposed algorithm is evaluated on MOT17 and MOT20 datasets and it achieves state-of-the-art performance on MOT17 and comparable results on MOT20. Also for evaluating the performance of the camera motion removal block, the SportsMOT dataset is used which has dynamic camera motion.

<p>
<img align="center" width="80%" src="https://github.com/mhnasseri/for_tracking/blob/main/pics/Block_Diagram_NS.jpg">
</p>

## Dependencies

The code is compatible with Python 3. The following dependencies are needed to run the tracker:

* NumPy
* sklearn
* OpenCV
* filterpy
* scipy
* motmetric
* loguru

## Installation

First, clone the repository:
```
git clone https://github.com/mhnasseri/for_tracking.git
```
Then, download the MOT17 dataset from [here](https://motchallenge.net/data/MOT17/) and the MOT20 dataset from [here](https://motchallenge.net/data/MOT20/). The detections are extracted as explained in the "[ByteTrack](https://arxiv.org/abs/2110.06864)" paper using [YOLOX](https://arxiv.org/abs/2107.08430) algorithm.

These detections are placed in the detections folder.

## Running the tracker

To run the code, at the beginning of __main__ in the __tracker_app.py__ file, you should specify the `phase` and `dataset` variables. The `phase` variable could be `train` or `test` and the dataset could be `MOT17` or `MOT20`. After that, the address of where each dataset is placed should be put in the `mot_path` variable. Also, the path to the outputs of the algorithm for each dataset should be placed in the `outFolder` variable. Besides, the address to the image and video outputs of the algorithm for each dataset should be written in the `imgFolder` variable.

In addition, the important variables in the algorithm are specified for each sequence in two datasets and can be changed to evaluate their effect on the results. 

## Evaluating the Results
 To evaluate the results, the `motmetric` library is used. If `phase == train`, after running the algorithm and generating the results, the results are evaluated by comparing them to the ground truth. The different metrics such as MOTA, IDF1, MOTP, FP, FN, IDS, FM, ML, and MT are calculated and shown in the terminal. These metrics are also saved in a text file in the specified folder in the `outFolder`.

## Main Results

The results of running the proposed algorithm on each train and each test sequence of MOT17 are shown in the two tables below. The private detections from ByteTrack paper which are extracted by [YOLOX](https://arxiv.org/abs/2107.08430) algorithm are used.

Sequence| MOTA | MOTP | IDF1 | MT | ML | FP | FN | IDS | FM 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----
MOT17-02| 82.2 | 84.2 | 69.7 | 44 | 3 | 945 | 2238 | 122 | 202
MOT17-04| 98.8 | 91.3 | 94.8 | 83 | 0 | 151 | 416 | 22 | 37
MOT17-05| 82.7 | 84.4 | 78.6 | 86 | 7 | 360 | 775 | 64 | 125
MOT17-09 | 92.5 | 86.7 | 85.3| 25 | 0 | 76 | 314 | 10 | 34
MOT17-10| 82.8 | 80.7 | 76.0 | 46 | 0 | 562 | 1571 | 78 | 264
MOT17-11| 91.5 | 87.9 | 88.5 | 62 | 2 | 306 | 478 | 20 | 60 
MOT17-13| 88.8 | 82.3 | 83.1 | 93 | 5 | 255 | 937 | 109 | 122
ALL| 91.3 | 87.3 | 85.4 | 439 | 17 | 2655 | 6729 | 425 | 844

Sequence| MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDS | FM 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----
MOT17-01| 63.9| 71.0 | 56.3 | 13 | 5 | 528 | 1785 | 17 | 75
MOT17-03| 92.5 | 87.3 | 70.9 | 141 | 0 | 3515 | 4258 | 91 | 257
MOT17-06| 63.2 | 67.4 | 52.5 | 112 | 41 | 1292 | 2897 | 151 | 209
MOT17-07 | 74.0 | 65.2 | 53.4 | 32 | 2 | 738 | 3566 | 86 | 202
MOT17-08| 64.7 | 56.4 | 48.9 | 40 | 6 | 1612 | 5658 | 197 | 360
MOT17-12| 65.8 | 73.9  | 60.8 | 42 | 12 | 812 | 2117 | 36 | 125
MOT17-14| 59.7 | 63.8 | 48.7 | 56 | 19 | 1061 | 6203 | 188 | 329
ALL| 80.4 | 77.7 | 63.3 | 1308 | 255 | 28674 | 79452 | 2298 | 4671

In addition, the result of the proposed algorithm on MOT17 alongside other algorithms is presented in the following table. 

Tracker | MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDS | FM | FPS 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|----------
[FairMOT](https://arxiv.org/abs/2004.01888) | 73.7 | 72.3 | 59.3 | 1017 | 408 | 27507 | 117477 | 3303 | 8073 | 25.9 
[GCNNMatch](https://arxiv.org/abs/2010.00067) | 57.3 | 56.3 | 45.4 | 575 | 787 | **14100** | 225042 | **1911** | 2837 | 1.3 
[GSDT](https://arxiv.org/abs/2006.13164) | 73.2 | 66.5 | 55.2 | 981 | 411 | 26397 | 120666 | 3891 | 8604 | 4.9 
[Trackformer](https://arxiv.org/abs/2101.02702) | 74.1 | 68.0 | 57.3 | 1113 | 246 | 34602 | 108777 | 2829 | 4221 | 5.7 
[TransTrack](https://arxiv.org/abs/2012.15460) | 75.2 | 63.5 | 54.1 | 1302 | **240** | 50157 | 86442 | 3603 | 4872 | 59.2 
[TransCenter](https://arxiv.org/abs/2103.15145) | 73.2 | 62.2 | 54.5 | 960 | 435 | 23112 | 123738 | 4614 | 9519 | 1.0 
[TransMOT](https://arxiv.org/abs/2104.00194) | 76.7 | 75.1 | 61.7 | 1200 | 387 | 36231 | 93150 | 2346 | 7719 | 1.1 
[ReMOT](https://www.sciencedirect.com/science/article/abs/pii/S0262885620302237) | 77.0 | 72.0 | 59.7 | 1218 | 324 | 33204 | 93612 | 2853 | 5304 | 1.8 
[ByteTrack](https://arxiv.org/abs/2110.06864) | 80.3 | 77.3 | 63.1 | 1254 | 342 | 25491 | 83721 | 2196 | 2277 | 29.6 
[OCSORT](https://arxiv.org/abs/2203.14360) | 78.0 | 77.5 | 63.2 | 966 | 492 | 15129 | 107055 | 1950 | **2040** | 29.0 
[FOR_Tracking](https://www.sciencedirect.com/science/article/pii/S104732032200270X) | **80.4** | **77.7** | **63.3** | **1311** | 255 | 28887 | **79329** | 2325 | 4689 | **60.7** 

You can see the metrics and the resultant video of each test sequence of the MOT17 dataset from [this link](https://motchallenge.net/method/MOT=5839&chl=10).

The proposed algorithm was evaluated on the MOT20 dataset, too. The result of the running algorithm on each train and each test sequence of this dataset are coming in the tables below.

Sequence| MOTA | MOTP | IDF1 | MT | ML | FP | FN | IDS | FM 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----
MOT20-01| 93.1 | 86.8 | 87.9 | 69 | 0 | 261 | 1062 | 40 | 77
MOT20-02| 91.4 | 86.4 | 77.4 | 254 | 1 | 5177 | 7739 | 198 | 14
MOT20-03| 94.6 | 84.3 | 94.5 | 626 | 23 | 4274 | 12290 | 114 | 16
MOT20-05 | 93.2 | 85.5 | 89.2| 1077 | 27 | 25494 | 17616 | 455 | 39
ALL| 93.4 | 85.3 | 89.0 | 2026 | 51 | 35206 | 38707 | 1493 | 2638

Sequence| MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDS | FM 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----
MOT20-04| 87.1| 84.4 | 67.8 | 576 | 12 | 16040 | 18703 | 512 | 1161
MOT20-06| 65.2 | 64.0 | 52.1 | 134 | 72 | 5116 | 40560 | 524 | 994
MOT20-07| 83.1 | 77.8 | 65.4 | 90 | 3 | 1539 | 3925 | 119 | 183
MOT20-08 | 57.7 | 64.0 | 48.6 | 57 | 46 | 4417 | 28066 | 288 | 718
ALL| 76.8 | 76.4 | 61.4 | 857 | 133 | 27112 | 91254 | 1443 | 3056

The result of the algorithm in comparison to other algorithms on the MOT20 dataset is coming in the table below.

Tracker | MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDS | FM | FPS 
----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|----------
[FairMOT](https://arxiv.org/abs/2004.01888) | 61.8 | 67.3 | 54.6 | 855 | **94** | 103440 | 88901 | 5243 | 7874 | 13.2 
[GCNNMatch](https://arxiv.org/abs/2010.00067) | 54.5 | 49.0 | 40.2 | 407 | 317 | **9522** | 223611 | 2038 | 2456 | 0.1 
[GSDT](https://arxiv.org/abs/2006.13164) | 67.1 | 67.5 | 53.6 | 660 | 164 | 31507 | 135395 | 3230 | 9878 | 1.5 
[Trackformer](https://arxiv.org/abs/2101.02702) | 68.6 | 65.7 | 54.7 | 660 | 181 | 20348 | 140373 | 1532 | 2474 | 5.7 
[TransTrack](https://arxiv.org/abs/2012.15460) | 65.0 | 59.5 | 48.9 | 622 | 1667 | 27191 | 150197 | 3608 | 11352 | 14.9 
[TransCenter](https://arxiv.org/abs/2103.15145) | 61.0 | 49.8 | 43.5 | 601 | 192 | 49189 | 147890 | 4493 | 8950 | 1.0 
[TransMOT](https://arxiv.org/abs/2104.00194) | 77.5 | 75.2 | 61.9 | **878** | 113 | 34201 | **80788** | 1615 | 2421 | 2.6 
[ReMOT](https://www.sciencedirect.com/science/article/abs/pii/S0262885620302237) | 77.4 | 73.1 | 61.2 | 846 | 123 | 28351 | 86659 | 1789 | 2121 | 0.4 
[ByteTrack](https://arxiv.org/abs/2110.06864) | **77.8** | 75.2 | 61.3 | 859 | 118 | 26249 | 87594 | 1223 | 1460 | 17.5 
[OCSORT](https://arxiv.org/abs/2203.14360) | 75.7 | 76.3 | **62.4** | 813 | 160 | 19067 | 105894 | **942** | **1086** | **18.7** 
[FOR_Tracking](https://www.sciencedirect.com/science/article/pii/S104732032200270X) | 76.8 | **76.4** | 61.4 | 857 | 133 | 27112 | 91254 | 1443 | 3056 | 15.6

Moreover, the evaluated metrics and the resultant video can be seen from [this link](https://motchallenge.net/method/MOT=5839&chl=13).

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, and visualize the tracker. The main entry point is in `tracker_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In the folder `libs` are written libraries that are used in tracking code:

* `basetrack.py`: Some basic definitions and functions.
* `calculate_cost.py`: Calculate association measures. There are functions for calculating different kinds of Intersection over Union(IoU) between two bounding boxes. (iou(), giou(), niou(), iou_ext(), iou_ext_sep(), ios()(Covered Percent) functions) 
* `calculate_fn_fp.py`: Calculate false negatives and false positives using the ground-truth files.
* `convert.py`: convert the format of a bounding box from [x1, y1, x2, y2] to [x, y, s, r] and vice versa.
* `for_tracker.py`: The high-level implementation of the "Online relational tracking with camera motion suppression" algorithm. For every sequence, a FOR tracker is initialized. It keeps track of different parameters of the tracker, such as targets and unmatched detections. At first, the location of all targets in the previous frame is predicted using the specific Kalman filter of every target. Then targets are associated in a cascade manner. At first, detections are matched with high-score detections. Then the unmatched tracklets are matched with low score detections with more cautions. After matching, the camera motion is calculated and the target estimated bounding boxes are updated and the cascade matching is repeated. Then a relational model is used to detect occluded tracklets from remaining unmatched tracklets. In the end, some remaining unmatched tracklets are removed and new targets are initialized from unmatched detections with high scores.
* `kalman_tracker.py`: Initialize a Kalman Filter for every new target. Then predict its location and correct its estimation A Kalman filter with a constant velocity model is initialized for every new target. (KalmanBoxTracker()) This filter is used for predicting the location of the target in the new frame. (predict()) The predicted location is corrected if it is matched with a detection. (update())
* `matching.py`: Associate detections to tracklets according to the association matrix and apply a minimum similarity on association. 
* `visuallization.py`: Used for drawing bounding boxes of targets and detections in every frame and generating video for every sequence. By changing DisplayState class variables, drawing different bounding boxes is enabled or disabled. The detections are drawn with a thin black rectangle. The targets are drawn with colored thick rectangles. The extended bounding boxes are shown with dashed-colored rectangles and ground truths are shown with thin red rectangles.   

The `tracker_app.py` expects detections in motchallenge format

## Citing (Fast) Online Relational Tracking (FOR Tracking) 

If you find this repo useful in your research, please consider citing the following paper:


    @article{nasseri2022online,
      title={Online relational tracking with camera motion suppression},
      author={Nasseri, Mohammad Hossein and Babaee, Mohammadreza and Moradi, Hadi and Hosseini, Reshad},
      journal={Journal of Visual Communication and Image Representation},
      pages={103750},
      year={2022},
      publisher={Elsevier}
    }
