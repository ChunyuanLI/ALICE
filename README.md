# ALICE
Adversarially Learned Inference with Conditional Entropy (ALICE)

In unsupervised learning case, two variants of ALICE are proposed to bound the conditional entropies, inclduing (1) explicit L2-norm reconstruction (ALICE_l2.py), and (2) Adversarially learned reconstruction (ALICE_A.py).


    python ALICE_l2.py
    
    python ALICE_A.py
    
    

One may tune the weighting hyperparameters $\lambda$ for the reconstruction terms for better performance. Note that ALICE reduces to ALI when $\lambda=0$.

TODO:
Add code for supervised learning.
