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

## Toy dataset

In *unsupervised learning* case, the setting of z ~ 1-GMM and x ~ 5-GMM is considered, two variants of ALICE are proposed to bound the conditional entropies, inclduing 

- (a) Explicitly specified L2-norm cycle-consistency (`ALICE_l2.py`) 
- (b) Implicitly learned cycle-consistency via adversarial training (`ALICE_A.py`).

In *weakly-supervised learning* case, the setting of z ~ 2-GMM and x ~ 5-GMM is considered, we only provide correspondences from 5 pairs of (x,z) of inverse spatial location relation. Two variants of ALICE are proposed to leverage the supervised information, including 

- (c) Explicitly specified L2-norm mapping  (`ALICE_l2_l2.py`) 
- (d) Implicitly learned mapping via adversarial training (`ALICE_A_A.py`) 

 
 <img src="/toy_data/results/toy_data_results.png" width="700px" />
 
    
One may tune the weighting hyperparameters of CE regularizers (cycle-consistency and/or supervised mappings) for better performance. Note that ALICE reduces to ALI when the weighting hyperparameters are 0.

## Reproduce figures in the paper

`plot_generation/alice_plots.ipynb`

## Real datasets

TODO
### MNIST
We study the impact of weighting hyperparameter (\lambda) for CE regularizer. The performance of image generation is evaluated by **inception score (ICP)**, and image reconstruction is evaluted by **mean square error (MSE)**.

Best ICP=9.279+-0.07, and MSE=0.0803+-0.007, when \lambda=1

Image Generation             |  Image Reconstruction
:-------------------------:|:-------------------------:
![](/plot_generation/figures/mnist_icp_weighting.png)  |  ![](/plot_generation/figures/mnist_mse_weighting.png)

### CIFAR

Best ICP=6.015+-0.0284, and MSE=0.4155+-0.2015, when \lambda=1e-6. Larger \lambda leads to lower MSE.

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
