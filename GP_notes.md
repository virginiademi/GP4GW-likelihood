In this project we employ Deep Kernel Learning, using:
* a Neural Network made up of 3 layers with 128 nodes each (`DeepKernel` class defined in `gp4gw/deep_learning.py`)
* a base kernel constructed by combining Matern1/2, RBF and Periodic kernel (specified in notebook and passed to `DeepKernel` class)

# Resources on Kernel Architecture and Inference Scheme
* **Kernel cookbook** [here](https://peterroelants.github.io/posts/gaussian-process-kernels/)
* **GP Sparse Variational Derivation**: [GPFlow docs](https://gpflow.readthedocs.io/en/master/notebooks/theory/SGPR_notes.html)
* **Sparse Variational GP for non-gaussian problems**:
[Towardsdatascience blog](https://towardsdatascience.com/variational-gaussian-process-what-to-do-when-things-are-not-gaussian-41197039f3d4)
* **Sparse Variational GP for large data sets**:
[Towardsdatascience blog](https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7)
* **Deep kernel learning**: 
* [Wilson 2016](http://proceedings.mlr.press/v51/wilson16.pdf)
* [Promises and Pitfalls of DKL - Rasmussen 2021](https://arxiv.org/pdf/2102.12108.pdf)

Non-stationary kernel is able to learn different rate of change, the higher the correlation the more the shape is distorted. <br/>
For this application, we don't want predicted distribution to be constrained, we want to memorise the data.

* **GPSS2020**: Deep Gaussian Processes explained by Neil Lawrence [here](http://inverseprobability.com/talks/notes/deep-gps.html)

* **Similar work**

15D GW deep learning [Green 2020](https://arxiv.org/pdf/2008.03312.pdf) <br/>
Reconstructing probability distributions with GPs [McClintock 2019](https://arxiv.org/abs/1905.09299)
Active Learning Accelerated Bayesian Inference [ALABI - J.Birky](https://github.com/jbirky/alabi)

* Effects of varying kernel hyper-parameters: lenghtscale, variance, likelihood noise
 ![image](https://user-images.githubusercontent.com/32883514/144617566-ffaa4ab2-a4ca-4090-9a59-21b849664e1b.png)
