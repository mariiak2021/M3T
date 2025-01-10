
# TrackEval
*Code for evaluating object tracking.*

This codebase provides code for a number of different tracking evaluation metrics (including the [HOTA metrics](https://link.springer.com/article/10.1007/s11263-020-01375-2)), as well as supporting running all of these metrics on a number of different tracking benchmarks. Plus plotting of results and other things one may want to do for tracking evaluation.

## Official Evaluation Code

The following benchmarks use TrackEval as their official evaluation code, check out the links to see TrackEval in action:

 -
 - **[MOTChallenge](https://motchallenge.net/)** ([Official Readme](docs/MOTChallenge-Official/Readme.md))
 


## Currently implemented metrics

The following metrics are currently implemented:

Metric Family | Sub metrics | Paper | Code | Notes |
|----- | ----------- |----- | ----------- | ----- |
| | | |  |  |
|**HOTA metrics**|HOTA, DetA, AssA, LocA, DetPr, DetRe, AssPr, AssRe|[paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)|[code](trackeval/metrics/hota.py)|**Recommended tracking metric**|
|**CLEARMOT metrics**|MOTA, MOTP, MT, ML, Frag, etc.|[paper](https://link.springer.com/article/10.1155/2008/246309)|[code](trackeval/metrics/clear.py)| |
|**Identity metrics**|IDF1, IDP, IDR|[paper](https://arxiv.org/abs/1609.01775)|[code](trackeval/metrics/identity.py)| |


## HOTA metrics

This code is also the official reference implementation for the HOTA metrics:

*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*

HOTA is a novel set of MOT evaluation metrics which enable better understanding of tracking behavior than previous metrics.

For more information check out the following links:
 - [Short blog post on HOTA](https://jonathonluiten.medium.com/how-to-evaluate-tracking-with-the-hota-metrics-754036d183e1) - **HIGHLY RECOMMENDED READING**
 - [IJCV version of paper](https://link.springer.com/article/10.1007/s11263-020-01375-2) (Open Access)
 - [ArXiv version of paper](https://arxiv.org/abs/2009.07736)
 - [Code](trackeval/metrics/hota.py)


## Running the code

The code can be run in one of two ways:

 - From the terminal via one of the scripts [here](scripts/). See each script for instructions and arguments, hopefully this is self-explanatory.
 - Directly by importing this package into your code, see the same scripts above for how. 


## Evaluate on your own custom benchmark

To evaluate on your own data, you have two options:
 - Write custom dataset code (more effort, rarely worth it).
 - Convert your current dataset and trackers to the same format of an already implemented benchmark.

To convert formats, check out the format specifications defined [here](docs).

By default, we would recommend the MOTChallenge format, although any implemented format should work. Note that for many cases you will want to use the argument ```--DO_PREPROC False``` unless you want to run preprocessing to remove distractor objects.

## Requirements
 Code tested on Python 3.7.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
 - For DAVIS dataset: Pillow
 - For J & F metric: opencv_python, scikit_image
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

use ```pip3 -r install minimum_requirments.txt``` to only install the minimum if you don't need the extra functionality as listed above.


## License

TrackEval is released under the [MIT License](LICENSE).

## Contact

If you encounter any problems with the code, please contact [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) ([luiten@vision.rwth-aachen.de](mailto:luiten@vision.rwth-aachen.de)).
If anything is unclear, or hard to use, please leave a comment either via email or as an issue and I would love to help.

## Dedication

This codebase was built for you, in order to make your life easier! For anyone doing research on tracking or using trackers, please don't hesitate to reach out with any comments or suggestions on how things could be improved.

## Contributing

We welcome contributions of new metrics and new supported benchmarks. Also any other new features or code improvements. Send a PR, an email, or open an issue detailing what you'd like to add/change to begin a conversation.

## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```BibTeX
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

Furthermore, if you use the HOTA metrics, please cite the following paper:

```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

If you use any other metrics please also cite the relevant papers, and don't forget to cite each of the benchmarks you evaluate on.
