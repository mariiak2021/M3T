# M3T: Multi-class Multi-instance Multi-view Object Tracking for Embodied AI Tasks 
<p>Mariia Khan⋆†, Jumana Abu-Khalaf⋆, David Suter⋆, Bodo Rosenhahn†, </p>


<p>⋆† School of Science, Edith Cowan University (ECU), Australia</p>

<p>†Institute for Information Processing, Leibniz University of Hannover (LUH), Germany</p>

[[`Paper`](https://link.springer.com/chapter/10.1007/978-3-031-25825-1_18)] - accepted to [IVCNZ22](https://ivcnz2022.aut.ac.nz/)

<p float="left">
  <img src="m3tmain.jpg?raw=true" width="85%" /> 
</p>

In this paper, we propose an extended multiple object tracking (MOT) task definition for embodied AI visual exploration research task - multi-class, multi-instance and multi-view object tracking (M3T). The aim of the proposed M3T task is to identify the unique number of objects in the environment, observed on the agent’s way, and visible from far or close view, from different angles or visible only partially. Classic MOT algorithms are not applicable for the M3T task, as they typically target moving single-class multiple object instances in one video and track objects, visible from only one angle or camera viewpoint. Thus, we present the M3T-Round algorithm designed for a simple scenario, where an agent takes 12 image frames, while rotating 360° from the initial position in a scene. We, first, detect each object in all image frames and then track objects (without any training), using cosine similarity metric for association of object tracks. The detector part of our M3T-Round algorithm is compatible with the baseline YOLOv4 algorithm in terms of detection accuracy: a 5.26 point improvement in AP75. The tracker part of our M3T-Round algorithm shows a 4.6 point improvement in HOTA over GMOTv2 algorithm, a recent, high-performance tracking method. Moreover, we have collected a new challenging tracking dataset from AI2-Thor simulator for training and evaluation of the proposed M3T-Round algorithm.

## News
The code for training, testing and evaluation of M3T-Round model is released on 10.01.25. 

The M3T dataset will be released shortly.

## 💻 Installation

To begin, clone this repository locally
```bash
git clone git@github.com:mariiak2021/SAOMv1-SAOMv2-and-PanoSAM.git 
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `TrackEval` the folder for evaluation of the tracker part of the algorithm;
* `requirements.txt` the file with all requirements to set up environment
* `sort_features_last_YOLO.py` the file` used for getting the tracker results for the M3T-Round model
* `create_MOT_anno.py` the file to create MOT format annotations
* `yolo.py` the file to get detections in YOLOv4 format for all images in the test set
* `obj1_416.names` the file with object lasses supported

</p>


You can install requirements by running
```bash
pip3 -r install requirements.txt
```

</p>
</details>

<p>
The model doesn't need any training. Just get the detection results with any detection algorithm like YOLov4, then get tracking results based on the detection:
```bash
sort_features_last_YOLO.py
```

Place tracking results to the folder .../M3T/TrackEval/data/trackers/mot_challenge/ai2thor-all/detection/data. 

To evaluate the model run:
```bash
cd TrackEval/scripts/
python run_mot_challenge.py
```

</p>
