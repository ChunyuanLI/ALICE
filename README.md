# ALICE
Adversarially Learned Inference with Conditional Entropy (**ALICE**)

[ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching](https://arxiv.org/abs/1709.01215)  
 [Chunyuan Li](http://chunyuan.li/),
 [Hao Liu](https://hliu96.github.io/), 
 [Changyou Chen](https://www.cse.buffalo.edu/~changyou/), 
 [Yunchen Pu](https://scholar.google.com/citations?user=ftW7RoAAAAAJ&hl=en), 
 [Liqun Chen](https://scholar.google.com/citations?user=T9T8Il0AAAAJ&hl=en), 
 [Ricardo Henao](https://scholar.google.com/citations?user=p_mm4-YAAAAJ),
 [Lawrence Carin](http://people.ee.duke.edu/~lcarin/)  
 Duke University. NIPS, 2017.

![](/plot_generation/figures_alice/alice_log_movie.gif)

*Alice4Alice: ALICE algorithms for painting the cartton of Alice's Adventures in Wonderland*

## Four variants of ALICE on toy dataset
In *unsupervised learning* case: 

- (a) Explicit cycle-consistency ([`ALICE_l2.py`](/toy_data/ALICE_l2.py)) 
- (b) Implicit cycle-consistency ([`ALICE_A.py`](/toy_data/ALICE_A.py))

In *weakly-supervised learning* case:

- (c) Explicit mapping  ([`ALICE_l2_l2.py`](/toy_data/ALICE_l2_l2.py)) 
- (d) Implicit mapping  ([`ALICE_A_A.py`](/toy_data/ALICE_A_A.py)) 

## Reproduce figures in the paper

[`plot_generation/alice_plots_paper.ipynb`](./plot_generation/alice_plots_paper.ipynb)

## Real datasets

TODO
### MNIST
We study the impact of weighting hyperparameter (\lambda) for CE regularizer. The performance of image generation is evaluated by **inception score (ICP)**, and image reconstruction is evaluted by **mean square error (MSE)**.

Best ICP=9.279 ± 0.07, and MSE=0.0803 ± 0.007, when \lambda=1

Note: we pre-trained a "perfect" MNIST classifier (100\% training accuracy) to compute the [inception score for MNIST](https://github.com/ChunyuanLI/MNIST_Inception_Score).

Image Generation             |  Image Reconstruction
:-------------------------:|:-------------------------:
![](/plot_generation/figures/mnist_icp_weighting.png)  |  ![](/plot_generation/figures/mnist_mse_weighting.png)

### CIFAR

Best ICP=6.015 ± 0.0284, and MSE=0.4155 ± 0.2015, when \lambda=1e-6. Larger \lambda leads to lower MSE.

Note: The quality of generated cifar images is evaluated via the [inception score based on ImageNet](https://github.com/openai/improved-gan/tree/master/inception_score)

Image Generation             |  Image Reconstruction
:-------------------------:|:-------------------------:
![](/plot_generation/figures/cifar_icp_weighting.png)  |  ![](/plot_generation/figures/cifar_mse_weighting.png)


### CelebA
### Car2Car
### Edge2Shoes


## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1709.01215):

```
@article{li2017alice,
  title={ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching},
  author={Li, Chunyuan and Liu, Hao and Chen, Changyou and Pu, Yunchen and Chen, Liqun and Henao, Ricardo and Carin, Lawrence},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```
