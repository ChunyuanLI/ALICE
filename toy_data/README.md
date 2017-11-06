## Toy dataset

### Impact of CE regularized

Best ICP=4.595 ± 0.604, and MSE=0.022 ± 0.029, when \lambda=1

Note: we pre-trained a "perfect" toy data classifier (100\% training accuracy) to compute the [inception score for our 5-GMM toy dataset](https://github.com/ChunyuanLI/MNIST_Inception_Score).

Image Generation             |  Image Reconstruction
:-------------------------:|:-------------------------:
![](/plot_generation/figures/toy_icp_weighting.png)  |  ![](/plot_generation/figures/toy_mse_weighting.png)


### Four algorithms

In *unsupervised learning* case, the setting of z ~ 1-GMM and x ~ 5-GMM is considered, two variants of ALICE are proposed to bound the conditional entropies, inclduing 

- (a) Explicitly specified L2-norm cycle-consistency ([`ALICE_l2.py`](/toy_data/ALICE_l2.py)) 
- (b) Implicitly learned cycle-consistency via adversarial training ([`ALICE_A.py`](/toy_data/ALICE_A.py))

In *weakly-supervised learning* case, the setting of z ~ 2-GMM and x ~ 5-GMM is considered, we only provide correspondences from 5 pairs of (x,z) of inverse spatial location relation. Two variants of ALICE are proposed to leverage the supervised information, including 

- (c) Explicitly specified L2-norm mapping  ([`ALICE_l2_l2.py`](/toy_data/ALICE_l2_l2.py)) 
- (d) Implicitly learned mapping via adversarial training ([`ALICE_A_A.py`](/toy_data/ALICE_A_A.py)) 

 (a) Explicit Cycle-Consistenty  |  (b) Implicit Cycle-Consistenty  
:-------------------------:|:-------------------------:
![width="425"](/toy_data/results/l2_results.png)|![width="425"](/toy_data/results/A_results.png)

 (c) Explicit Mapping  |  (d) Implicit Mapping
:-------------------------:|:-------------------------:
![](/toy_data/results/l2_l2_results.png)|![](/toy_data/results/A_A_results.png)
 
    
One may tune the weighting hyperparameters of CE regularizers (cycle-consistency and/or supervised mappings) for better performance. Note that ALICE reduces to ALI when the weighting hyperparameters are 0.
