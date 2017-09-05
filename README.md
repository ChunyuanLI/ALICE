# ALICE
Adversarially Learned Inference with Conditional Entropy (ALICE)

In unsupervised learning case, two variants of ALICE are proposed to bound the conditional entropies, inclduing 

(1) explicit L2-norm reconstruction (ALICE_l2.py) 

    python ALICE_l2.py
    
(2) Adversarially learned reconstruction (ALICE_A.py).

    python ALICE_A.py
    
    

One may tune the weighting hyperparameters <img src="https://latex.codecogs.com/gif.latex?$\lambda$" /> for the reconstruction terms for better performance. Note that ALICE reduces to ALI when <img src="https://latex.codecogs.com/gif.latex?$\lambda=0$" />.

TODO:
Add code for the two variants of ALICE in the supervised learning case, including explicit l2 regression loss, and adversarially learned mapping.
