import numpy as np
import arviz as az
import tensorflow as tf
from gpflow.config import default_float
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_data(x, y, split_ratio=0.8):
    """
    Splitting training and testing data
    Normalising x-data according to MinMaxScaler
    Normalising y-data according to StandardScaler
    :param x: needs to have shape (n_samples, n_dim)
    :param y: needs to have shape (n_samples, 1)
    :param split_ratio: default is 0.8
    :return: dictionary of training data, test data and
    corresponding scaling functions
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    assert x.dtype == y.dtype
    dtype = x.dtype
    N, Q = x.shape
    N_split = int(np.ceil(split_ratio * N))
    
#     tails = y.max() - y.max() * 0.2
#     x_tails = x[y < tails, :]
#     y_tails = y[y < tails]
#     assert len(x_tails) == len(y_tails)
#     N_train = N_split - len(x_tails)
    
#     x_train = np.concatenate((x_tails, x[y > tails, :][:N_train, :]), axis = 0)
#     x_test = x[y > tails, :][N_train:, :]

#     y_train = np.concatenate((y_tails, y[y > tails][:N_train]))
#     y_test = y[y > tails][N_train:]
    x_train = x[:N_split,:]
    x_test = x[N_split:, :]
    
    y_train = y[:N_split]
    y_test = y[N_split:]

    y_scaler = StandardScaler()
    x_scaler = MinMaxScaler()
    y_scaler.fit(y_train.reshape([-1, 1]))
    x_scaler.fit(x_train)

    y_train_scaled = y_scaler.transform(y_train.reshape([-1, 1])).astype(dtype).reshape([-1])
    x_train_scaled = x_scaler.transform(x_train).astype(dtype)
    y_test_scaled = y_scaler.transform(y_test.reshape([-1, 1])).astype(dtype).reshape([-1])
    x_test_scaled = x_scaler.transform(x_test).astype(dtype)
    print('Shape of x-data: N=%.f, Q=%.f' % (N, Q))
    return {
    "x_train": x_train_scaled,
    "y_train": y_train_scaled,
    "x_test": x_test_scaled,
    "y_test": y_test_scaled,
    "y_scaler": y_scaler,
    "x_scaler": x_scaler
    }

def scale_data(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    assert x.dtype == y.dtype
    dtype = x.dtype
    N, Q = x.shape
    y_scaler = StandardScaler()
    x_scaler = MinMaxScaler()
    y_scaler.fit(y.reshape([-1, 1]))
    x_scaler.fit(x)
    
    y_scaled = y_scaler.transform(y.reshape([-1, 1])).astype(dtype).reshape([-1])
    x_scaled = x_scaler.transform(x).astype(dtype)
    print('Shape of x-data: N=%.f, Q=%.f' % (N, Q))
    return {
    "x_train": x_scaled,
    "y_train": y_scaled,
    "y_scaler": y_scaler,
    "x_scaler": x_scaler
    }



def convert_to_tf_and_downsample(my_data, n_train):
    """
    Returns X_train, Y_train, X_test, Y_test as
    TensorFlow objects, where the train data is also
    downsampled for faster training.
    :my_data: dataset scaled and divided in train/test with prepare_data()
    :param n_train: number of training points
    """
    N = my_data["x_train"].shape[0]
    Q = my_data["x_train"].shape[1]

    indices = np.random.randint(low=0, high=N, size=n_train)
    x_train_subset = my_data["x_train"][indices]
    y_train_subset = my_data["y_train"][indices]
    
    X_train = tf.convert_to_tensor(x_train_subset, dtype=default_float())
    Y_train = tf.convert_to_tensor(y_train_subset.reshape(-1, 1), dtype=default_float())

    X_test = tf.convert_to_tensor(my_data["x_test"], dtype=default_float())
    Y_test = tf.convert_to_tensor(my_data["y_test"].reshape(-1, 1), dtype=default_float())

    data = (X_train, Y_train)
    assert [data[i].shape[0] == N for i in range(2)] and data[0].shape[1] == Q
    print('Selected %.f training point and %.f testing points'% (X_train.shape[0], X_test.shape[0]))
    return X_train, Y_train, X_test, Y_test
    

def convert_to_arviz_data(tf_samples, parameters_scaler, param_names):
    """
    Returns 'InferenceData' posterior object
    :param tf_samples: mcmc or hmc samples from tf.run_chain 
    :param parameters_scaler: scaler for samples
    :param param_names: list of parameter names of samples
    :return: arviz posterior sample object
    """
    new_samples = np.swapaxes(tf_samples, 0, 1)
    samples = []
    for i in range(len(new_samples)):
        chain = parameters_scaler.inverse_transform(new_samples[i])
        samples.append(chain)
    new_samples = np.array(samples)
    dims = {key: new_samples.T[i].T for i, key in enumerate(param_names)}
    data = az.from_dict(dims)
    return data

def pdf_tails_sampling(samples, min_samples=30, bins=10,):
    """
    Downsample pdf by selecting all samples in the tails.
    `min_samples` determines how many samples to take for each bin,
    if the bin contains less than min_samples they are all selected.
    """
    freq, interv = np.histogram(samples, bins = bins)
    l = list()
    for n in range(bins):
        if freq[n] < min_samples: 
            l.append(samples[(samples>=interv[n]) & (samples<= interv[n+1])])
            l.append(np.random.choice(samples[(samples>= interv[np.argmax(freq)]) & ( samples<= interv[np.argmax(freq)+1])], size=min_samples-freq[n], replace=False))
        else:
            l.append(np.random.choice(samples[(samples>=interv[n]) & (samples<= interv[n+1])], size=min_samples, replace=False))
    l = np.concatenate(l, axis=0)
    print('Selecting {} points'.format(len(l)))
    return l
    
def convert_to_tf_and_tails_sampling(my_data, n_train, bins = 10):
    
    N = my_data["x_train"].shape[0]
    Q = my_data["x_train"].shape[1]
    
    min_samples = int(n_train/bins)
    array_test = list()
    for dim in range(Q):
        samples = my_data["x_train"][:, dim]
        l = pdf_tails_sampling(samples, min_samples=min_samples, bins=bins)
        print(len(l))
        array_test.append(l)
    
    x_train_subset = np.stack(array_test, axis=1)
    y_train_subset = pdf_tails_sampling(my_data["y_train"], min_samples=min_samples, bins=bins)
    
    X_train = tf.convert_to_tensor(x_train_subset, dtype=default_float())
    Y_train = tf.convert_to_tensor(y_train_subset.reshape(-1, 1), dtype=default_float())

    X_test = tf.convert_to_tensor(my_data["x_test"], dtype=default_float())
    Y_test = tf.convert_to_tensor(my_data["y_test"].reshape(-1, 1), dtype=default_float())

    data = (X_train, Y_train)
    assert [data[i].shape[0] == N for i in range(2)] and data[0].shape[1] == Q
    print('Selected %.f training point and %.f testing points'% (X_train.shape[0], X_test.shape[0]))
    return X_train, Y_train, X_test, Y_test

def convert_to_tf_and_tails_sampling_likelihood(my_data, n_train, bins = 10):
    
    N = my_data["x_train"].shape[0]
    Q = my_data["x_train"].shape[1]
    
    min_samples = int(n_train/bins)
    y_train_subset = pdf_tails_sampling(my_data["y_train"], min_samples=min_samples, bins=bins)
    indices = np.concatenate([np.argwhere(my_data['y_train'] == i)[0] for i in y_train_subset], axis = 0)
    
    x_train_subset = my_data["x_train"][indices]
    
    X_train = tf.convert_to_tensor(x_train_subset, dtype=default_float())
    Y_train = tf.convert_to_tensor(y_train_subset.reshape(-1, 1), dtype=default_float())

    X_test = tf.convert_to_tensor(my_data["x_test"], dtype=default_float())
    Y_test = tf.convert_to_tensor(my_data["y_test"].reshape(-1, 1), dtype=default_float())

    data = (X_train, Y_train)
    assert [data[i].shape[0] == N for i in range(2)] and data[0].shape[1] == Q
    print('Selected %.f training point and %.f testing points'% (X_train.shape[0], X_test.shape[0]))
    return X_train, Y_train, X_test, Y_test