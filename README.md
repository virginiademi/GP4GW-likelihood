# GP4GW-likelihood
Gaussian Processes for Gravitational Waves Likelihood Reconstruction.

The likelihood surface is modelled with a Gaussian Process as a multi-dimensional function y = f(x), where y are the likelihood evaluations over parameter space (need to use at least all sampled dimensions) and x are MCMC posterior samples.

The GP model is built using `GPFlow` and `TensorFlow`.
We built a `gp4gw` package which allows to load and prepare GW data for the modelling step. The package also allows to sample the modelled likelihood multiplied by the analytical priors in order to obtain more posterior samples in any desired region of the parameter space.

## Set-up on Eyas cluster
We need to create two virtual environments:
* one for `TensorFlow-GPU`, to run the GP model construction
* one for regular `TensorFlow` (CPU), to run the sampling over the surrogate model

The following steps will have to be repeated twice (`TensorFlow-GPU` version in brackets):
```python
conda create -n tensorflow (tensorflow_gpu)
source activate tensorflow (tensorflow_gpu)
conda install tensorflow==2.4.1 (tensorflow_gpu==2.4.1)
conda install tensorflow_probability==0.11
conda install ipykernel 
python -m ipykernel install --user --name tensorflow (tensorflow_gpu)
```
Once the environments have been set-up, we install the dependencies required by `gp4gw` (again, this step needs to be repeated for both environments).
```
mv gp4gw
pip install -r requirements.txt
python setup.py develop
```
## Running notebooks
To run the notebooks you need to set-up a jupyter hub session.
On one terminal window run:
```bash
jupyter notebook --no-browser --port=8084 --debug
```
this command will initialise a server and you will have to copy paste the localhost url onto a new browser page.
To be able to actually start the server you need to open a new terminal window and run:
```bash
ssh -L 8084:localhost:8084 user.name@eyas.cf.ac.uk
```
The server should now start running and you should view the jupyter hub home page.

Once you open one of the notebooks, you should be able to see the kernels we created, one called `tensorflow` and one called `tensorflow_gpu`.
The `tensorflow_gpu` kernel is only needed to train the GP model, so if you are loading an existing model you can run everything with the `tensorflow` kernel.

**NB** Running two notebooks at the same time might cause Out Of Memory errors (OOM), so only work on one notebook at a time.

## Posterior samples
The posterior samples for GW150914 can be found on my own directory (file was too large to be uploaded to git), you can copy them by running:
```bash
cp ~/virginia.demilio/bilby_results_files/GW150914_data0_1126259462-391_analysis_H1L1_dynesty_merge_result.json .
```
Alternatively you can download the full set of posterior samples from the [GWTC-1 public release page](https://dcc.ligo.org/LIGO-P2000193/public). I downloaded the "Original bilby result files" using `wget` command.
