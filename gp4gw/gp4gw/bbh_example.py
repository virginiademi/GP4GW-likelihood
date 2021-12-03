import numpy as np
import pandas as pd
import math
import os
from itertools import compress
import bilby
from astropy import cosmology as cosmo
import tensorflow_probability as tfp
import tensorflow as tf

float_type = tf.float64

def load_data(path, param_names, samples_frac=0.5, dtype=np.float64):
    """
    Loading gravitational waves posterior samples for training and testing GP
    :param path: location of the files containing samples
    :param params_names: list of the parameters names of the samples
    :param samples_frac: fraction of samples that will be used (default is 0.5)
    :param dtype: default float64 (recommend not to change)
    :return: dictionary of x (posterior samples) and y (log likelihood)
    """
    extension = os.path.splitext(os.path.splitext(path)[1])[0].lstrip('.')
    if extension == 'dat':
        data = pd.read_csv(path).sample(frac=samples_frac)
    elif extension == 'json':
        bilby_object = bilby.result.read_in_result(path)
        data = bilby_object.posterior.sample(frac=samples_frac)
    else:
        raise Exception('{} data file not yet supported'.format(extension))
    try:
        target = data['log_likelihood']
    except KeyError:
        target = data['logl']
    x = np.array([data[x] for x in param_names], dtype=dtype).T
    y = np.array(target, dtype=dtype)
    return {
            "x": x,
            "y": y
            }

def priors_from_bilby_object(path):
    """
    Loads bilby's PriorDict from bilby result object
    :param path: path of bilby result file (json)
    :return: dictionary of bilby priors used
    """
    bilby_object = bilby.result.read_in_result(path)
    prior_dict = bilby_object.priors
#     if type(prior_dict['luminosity_distance'])==str and 'FlatLambdaCDM' in prior_dict['luminosity_distance']:
#         FlatLambdaCDM = cosmo.FlatLambdaCDM
#         prior_dict["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
#             minimum=100.0, 
#             maximum=5000.0, 
#             cosmology=FlatLambdaCDM(name="Planck15", H0=67.7, Om0=0.307, Tcmb0=2.725, Neff=3.05, Ob0=0.0486),
#             name='luminosity_distance', 
#             latex_label='$d_L$', 
#             unit='Mpc', 
#             boundary=None)
    return prior_dict

def load_bilby_prior(prior_file):
    """
    Loads bilby's PriorDict from prior file.
    :param prior_file: path to ".prior" file
    return: bilby's prior dictionary
    """
    prior_dict = bilby.core.prior.PriorDict()
    prior_dict.from_file(filename=prior_file)
    return prior_dict

def sampled_parameters(analytic_priors):
    """
    Extract list of sampled parameters from prior dictionary.
    :param analytic_priors: bilby's PriorDict
    :return: list of parameter names
    """
    all_keys = list(analytic_priors.keys())
    # Remove Constraint params
    not_constraint = [str(analytic_priors[key]).split('(')[0]!='Constraint' for key in all_keys]
    filtered_keys = list(compress(all_keys, not_constraint))
    # Remove fixed params
    not_delta_func = [str(analytic_priors[key]).split('(')[0]!='DeltaFunction' for key in filtered_keys]
    param_names = list(compress(filtered_keys, not_delta_func))
    return param_names

def scale_parameters(x, parameters_scaler):
    x*=parameters_scaler.scale_
    x+=parameters_scaler.min_
    return x

def quick_priors(x, param_names, analytic_priors, parameters_scaler=None):
    """
    :param x: scaled or non-scaled x-array
    :param param_names: list of parameter names
    :param analytic_priors: bilby's prior dictionary
    :return: dictionary with reduced prior ranges and bilby's Uniform distributions
    replacd by tensorflow_probability equivalents
    """
    low_bound={}
    up_bound={}
    if parameters_scaler==None:
        print('Loading non-scaled priors bounds')
        low_bound =  {key: tf.cast(analytic_priors[key].minimum , dtype = float_type) for key in param_names}
        up_bound =  {key: tf.cast(analytic_priors[key].maximum , dtype = float_type) for key in param_names}
#         low_bound =  {key: tf.cast(np.min(x[:,param_names.index(key)]) , dtype = float_type) for key in param_names}
#         up_bound =  {key: tf.cast(np.max(x[:,param_names.index(key)]) , dtype = float_type) for key in param_names}
    else:
        print(f'Scaling prior bounds using {parameters_scaler}')

        array_min = parameters_scaler.transform(
        np.array([analytic_priors[key].minimum for key in param_names]
                ).reshape(1, -1))
        array_max = parameters_scaler.transform(
        np.array([analytic_priors[key].maximum for key in param_names]
                ).reshape(1, -1))
        low_bound =  {key: tf.cast(array_min[0,i] , dtype = float_type) for i, key in enumerate(param_names)}
        up_bound =  {key: tf.cast(array_max[0,i], dtype = float_type) for i, key in enumerate(param_names)}

    quick_priors = {}
    for key in list(analytic_priors.keys()):
        try:
            if type(analytic_priors[key]) == bilby.core.prior.analytical.DeltaFunction:
                quick_priors.pop(key)

            if type(analytic_priors[key]) == bilby.core.prior.analytical.Uniform:
#                 print('Note: uniform prior has tight range based on X')
#                 quick_priors[key] = tfp.distributions.Uniform(
#                     low=tf.cast(np.min(x[:, param_names.index(key)]), dtype = float_type), high=tf.cast(np.max(x[:, param_names.index(key)]), dtype = float_type)
#                 )
                quick_priors[key] = tfp.distributions.Uniform(
                    low=low_bound[key], high=up_bound[key]
                )
                
            if type(analytic_priors[key]) == bilby.core.prior.base.Constraint:
                if parameters_scaler==None:
                    quick_priors[key] = tf_Constraint(
                        minimum=analytic_priors[key].minimum, maximum=analytic_priors[key].maximum
                )
                else:
                    print(f'Constraint on {key} has been scaled with {parameters_scaler}')
                    fake_array = np.zeros_like(x[0,:]).reshape(1,-1)
                    fake_array[0,0] = analytic_priors[key].minimum
                    fake_array[0,1] = analytic_priors[key].maximum
                    scaled_array = parameters_scaler.transform(fake_array.reshape(1,-1))
                    quick_priors[key] = tf_Constraint(
                        minimum=scaled_array[0,0], maximum=scaled_array[0,1]
                )
            if type(analytic_priors[key]) == bilby.core.prior.analytical.Sine:
                quick_priors[key] = tf_Sine(
                    minimum=low_bound[key], maximum=up_bound[key]
                )
            if type(analytic_priors[key]) == bilby.core.prior.analytical.Cosine:
                quick_priors[key] = tf_Cosine(
                    minimum=low_bound[key], maximum=up_bound[key]
                )
            if type(analytic_priors[key]) == bilby.core.prior.analytical.PowerLaw:
                quick_priors[key] = tf_PowerLaw(alpha=analytic_priors[key].alpha,
                    minimum=low_bound[key], maximum=up_bound[key]
                )
            if type(analytic_priors[key]) == bilby.gw.prior.UniformComovingVolume:
                raise NotImplementedError 
    #                 quick_prior[key] = tf_UniformComovingVolume(minimum=low_bound[key], maximum=up_bound[key])
        except KeyError:
            pass
    return quick_priors


class tf_Sine:
    def __init__(self, minimum=0, maximum=math.pi):
        self.minimum = tf.cast(minimum, dtype=float_type)
        self.maximum = tf.cast(maximum, dtype=float_type)
    
    def prob(self, x):
        x = tf.cast(x, dtype=float_type)
        is_in_prior_range = tf.math.logical_and(tf.math.greater_equal(x, self.minimum), tf.math.less_equal(x, self.maximum))
        prob = tf.math.sin(x) / 2 
        out_of_bounds = tf.zeros_like(prob, dtype=tf.float64)
        prob = tf.where(is_in_prior_range, prob, out_of_bounds)
        return prob
    
    def log_prob(self, x):
        return tf.math.log(tf.cast(self.prob(x), dtype=float_type))


class tf_Cosine:
    def __init__(self, minimum=-math.pi/2, maximum=math.pi/2):
        self.minimum = tf.cast(minimum, dtype=float_type)
        self.maximum = tf.cast(maximum, dtype=float_type)
        
    def prob(self, x):
        x = tf.cast(x, dtype=float_type)
        is_in_prior_range = tf.math.logical_and(tf.math.greater_equal(x, self.minimum), tf.math.less_equal(x, self.maximum))
        prob = tf.math.cos(x) / 2 
        out_of_bounds = tf.zeros_like(prob, dtype=tf.float64)
        prob = tf.where(is_in_prior_range, prob, out_of_bounds)
        return prob
    
    def log_prob(self, x):
        return tf.math.log(tf.cast(self.prob(x), dtype=float_type))


class tf_Constraint:
    def __init__(self, minimum, maximum):
        self.minimum = tf.cast(minimum, dtype=float_type)
        self.maximum = tf.cast(maximum, dtype=float_type)
        
    def prob(self, x):
        x = tf.cast(x, dtype=float_type)
        prob = tf.math.logical_and(tf.math.greater(x, self.minimum), tf.math.less(x, self.maximum))
        return prob
    
    def log_prob(self, x):
        return tf.math.log(tf.cast(self.prob(x), dtype=float_type))


class tf_PowerLaw:
    def __init__(self, alpha, minimum, maximum):
        self.alpha = tf.cast(alpha, dtype=tf.float64)
        self.minimum = tf.cast(minimum, dtype=tf.float64)
        self.maximum = tf.cast(maximum, dtype=tf.float64)
        
    def log_prob(self, x):
        
        x = tf.cast(x, dtype=tf.float64)
#         Should have an if statement for alpha==-1, need to code up differently in tf
#         minus_one = tf.constant(-1, dtype=float_type)
#         tf.cond(tf.math.equal(self.alpha, minus_one), normalising = 1. / tf.math.log(self.maximum / self.minimum)
        normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))
        is_in_prior_range = tf.math.logical_and(tf.math.greater_equal(x, self.minimum), tf.math.less_equal(x, self.maximum))
        normalising = tf.convert_to_tensor(normalising, dtype=tf.float64)
        log_prob = tf.math.add(self.alpha * tf.math.log(x), tf.math.log(normalising))
        out_of_bounds = tf.ones_like(log_prob, dtype=tf.float64) * - math.inf
        log_prob = tf.where(is_in_prior_range, log_prob, out_of_bounds)
        return log_prob
    
    def prob(self, x):
        prob = tf.math.exp(tf.cast(self.log_prob(x), dtype=tf.float64))
        prob = tf.where(tf.math.is_nan(prob), tf.ones_like(prob) * - math.inf, prob)
        return prob
