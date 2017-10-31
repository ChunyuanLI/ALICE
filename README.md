# ALICE
Adversarially Learned Inference with Conditional Entropy (ALICE)

In unsupervised learning case, two variants of ALICE are proposed to bound the conditional entropies, inclduing 

- Explicitly specified L2-norm cycle-consistency (`ALICE_l2.py`) 
- Implicitly learned cycle-consistency via adversarial training (`ALICE_A.py`).

In weakly-supervised learning case, we only have correspondences from 5 pairs of (x,z). Two variants of ALICE are proposed to leverage the supervised information, including 

- Explicitly specified L2-norm mapping  (`ALICE_l2_l2.py`) 
- Implicitly learned mapping via adversarial training (`ALICE_A_A.py`) 

 
 <img src="/toy_data/results/toy_data_results.png" data-canonical-src="/toy_data/results/toy_data_results.png" width="800" height="500" />
 
    
One may tune the weighting hyperparameters <img src="https://latex.codecogs.com/gif.latex?$\lambda$" /> for the reconstruction terms for better performance. Note that ALICE reduces to ALI when <img src="https://latex.codecogs.com/gif.latex?$\lambda=0$" />.


## Citation
If you use this code for your research, please cite our [paper](https://junyanz.github.io/CycleGAN/):

```
@article{li2017alice,
  title={ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching},
  author={Li, Chunyuan and Liu, Hao and Chen, Changyou and Pu, Yunchen and Chen, Liqun and Henao, Ricardo and Carin, Lawrence},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
```
