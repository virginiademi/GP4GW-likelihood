# Some resurces and notes

* **Kernel cookbook** in particular addition [here](https://peterroelants.github.io/posts/gaussian-process-kernels/)
* **GP Sparse Variational Derivation**: [here](https://gpflow.readthedocs.io/en/master/notebooks/theory/SGPR_notes.html)
* **Sparse Variational GP for non-gaussian problems**:
[here](https://towardsdatascience.com/variational-gaussian-process-what-to-do-when-things-are-not-gaussian-41197039f3d4)
* **Sparse Variational GP for large data sets**:
[here](https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7)
* **Deep kernel learning**: http://proceedings.mlr.press/v51/wilson16.pdf

Non-stationary kernel is able to learn different rate of change, the higher the correlation the more the shape is distorted. <br/>
For this application, we don't want predicted distribution to be constrained, we want to memorise the data.

* **GPSS2020**: Deep Gaussian Processes explained by Neil Lawrence [here](http://inverseprobability.com/talks/notes/deep-gps.html)

* **Papers on similar work**

15D deep learning [here](https://arxiv.org/pdf/2008.03312.pdf) <br/>
Neutra [here](https://arxiv.org/pdf/1903.03704.pdf)

### Some observations:

The kernel parameters (lenghtscale and variance) have the greatest effect over the accuracy of the interpolation compared to all other factors (kernel architecture, number of inducing points, number of training epochs etc). <br/>
Lenghtscale: describes how smooth a function is. Small lenghtscale means that the function values can change quickly. Large values characterise functions that change only slowly. It also determines how far can we reliably extrapolate from the training data. <br/>
Lower GP prior -> lower variance but higher lenghtscale 


