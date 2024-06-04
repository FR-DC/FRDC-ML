## FRRD-60: Consistency as a feature discriminator for Novelty Detection

Hypothesis: Consistency can be used as a measure to discriminate between normal
and novel data. In the case of FRDC, it's a metric to separate seen and unseen
tree-species. Seen data will have high consistency, while unseen data will have
low consistency.

Conclusion: It's not possible. Consistency is a measure of the similarity
of output distributions, upon different pertubations of the input. A simple
counter-example is the following: a black square in a white background.
The consistency of that image is always perfect given weak augmentations 
(flips). We also show that using CIFAR10 and noise datasets that the
Jenson-Shannon divergence goes against our hypothesis, sometimes yielding
higher consistency despite being Out-of-Distribution (OOD).

Author Discussion: We believe that the formulated hypothesis, while on the 
surface, plausible, requires more mathematical rigor to be proven.
