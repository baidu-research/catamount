## Beyond Human-Level Accuracy: Computational Challenges in Deep Learning [(PDF)](PPoPP_2019_Projecting_Deep_Learning_Hardware_Requirements_Final.pdf)

[Joel Hestness](https://arxiv.org/search/cs?searchtype=author&query=Hestness%2C+J), [Newsha Ardalani](https://arxiv.org/search/cs?searchtype=author&query=Ardalani%2C+N), [Gregory Diamos](https://arxiv.org/search/cs?searchtype=author&query=Diamos%2C+G)

__Abstract:__
> Deep learning (DL) research yields accuracy and product improvements from both model architecture changes and scale: larger data sets and models, and more computation. For hardware design, it is difficult to predict DL model changes. However, recent prior work shows that as dataset sizes grow, DL model accuracy and model size grow predictably. This paper leverages the prior work to project the dataset and model size growth required to advance DL accuracy beyond human-level, to frontier targets defined by machine learning experts. Datasets will need to grow 33–971×, while models will need to grow 6.6–456× to achieve target accuracies.

> We further characterize and project the computational requirements to train these applications at scale. Our characterization reveals an important segmentation of DL training challenges for recurrent neural networks (RNNs) that contrasts with prior studies of deep convolutional networks. RNNs will have comparatively moderate operational intensities and very large memory footprint requirements. In contrast to emerging accelerator designs, large-scale RNN training characteristics suggest designs with significantly larger memory capacity and on-chip caches.

The paper can also be found on the [ACM website here](https://dl.acm.org/citation.cfm?doid=3293883.3295710)

Comments: 14 pages, 12 figures, In Proceedings of the 24th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '19). ACM, Washington DC, USA

BibTex:
```
@inproceedings{hestness:deeplearningcomputational:ppopp:2019,
  author={Joel Hestness and Newsha Ardalani and Gregory Diamos},
  title={Beyond Human-Level Accuracy: Computational Challenges in Deep Learning},
  booktitle={Proceedings of the 24th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '19)},
  month={February},
  year={2019},
  organization={ACM},
}
```
