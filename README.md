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
* `finetuneSAM.py` the fine-tuning script, which can be used for any of SAOMv1, SAOMv2 or PanoSAM training;
* `environment.yml` the file with all requirements to set up conda environment
* `show.py the file` used for saving output masks during testing the model
* `testbatch.py` the file to use while testing the re-trained model performance
* `eval_miou.py` the file to use for evaluating the output masks
* `DSmetadataPanoSAM.json` the mapping between masksa and images for PanoSAM model DS
* `DSmetadataSAOMv1.json` the mapping between masksa and images for SAOMv1 model DS
  `DSmetadataSAOMv2.json` the mapping between masksa and images for SAOMv2 model DS
* `per_segment_anything/`
    - `automatic_mask_generator.py` - The file used for testing fine-tuned SAM version, where you can set all parameters like IoU threshold.
    - `samwrapperpano.py` - The file used for training the model, e.g. finding the location prior for each object and getting it's nearest neighbor from the point grid.
* `persamf/` - the foder for output of the testing/training stages
* `dataset/`
    - `SCDTrack2PhD.py` - The file used for setting up the dataset files for traing/testing/validation

</p>


You can then install requirements by using conda, we can create a `embclone` environment with our requirements by running
```bash
export MY_ENV_NAME=embclip-rearrange
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file environment.yml --name $MY_ENV_NAME
```

Download weights for the original SAM  model (ViT-H SAM model and ViT-B SAM model.) from here (place the download .ph file into the root of the folder): 
```bash
https://github.com/facebookresearch/segment-anything
```
</p>
</details>

<p>
To train the model on several GPUs run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetuneSAM.py --world_size 4
```

To evaluate the model run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetuneSAM.py --world_size 4  --eval_only
```

After you get the output masks for evaluation run:
```bash
eval_miou.py
```

To run the re-trained model in the everything mode run:
```bash
tesbatch.py
```
</p>
