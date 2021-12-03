# GPFlow training utils
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from typing import Tuple, Optional

def optimization_step(model: gpflow.models.SVGP,
                      batch: Tuple[tf.Tensor, tf.Tensor],
                      optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
                      natgrad_opt: gpflow.optimizers = gpflow.optimizers.NaturalGradient(0.09)):
    """
    Optimization step for checkpointing loop. Employs NaturalGradient for variational parameters and a tensorflow optimiser for all other variables.
    :param model: gpflow model to be optimised
    :param batch: batched (X, Y) data
    :param optimizer: Adam optimiser as default, can change learning rate
    :param natgrad_opt: natural gradient object from gpfloe, can change gamma
    """
    natgrad_opt.minimize(
        model.training_loss_closure(batch), var_list=[(model.q_mu, model.q_sqrt)]
    )
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def optimization_exact(model: gpflow.models.GPR,
                     optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01)):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss()
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def tf_dataset(batch_size, X, Y):
    """
    Create a TensorFlow train_dataset object to be passed to the
    checkpointing loop.
    :param batch_size: int size of each batch
    :param X: X_train as Tensor
    :param Y: Y_train as Tensor
    """
    N = X.shape[0]
    Q = X.shape[1]
    
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = N // 2

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    original_train_dataset = train_dataset
    train_dataset = (
        train_dataset.repeat()
        .prefetch(prefetch_size)
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(batch_size)
    )
    return train_dataset, original_train_dataset

def checkpointing_training_loop(
        model: gpflow.models.SVGP,
        batch_size: int,
        num_batches_per_epoch: int,
        train_dataset: Tuple[tf.Tensor, tf.Tensor],
        epochs: int,
        manager: tf.train.CheckpointManager,
        optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
        natgrad_opt: gpflow.optimizers = gpflow.optimizers.NaturalGradient(0.1),
        logging_epoch_freq: int = 10,
        epoch_var: Optional[tf.Variable] = None,
        step_var: Optional[tf.Variable] = None,
        exp_tag: str = 'test'
):
    """
    Loop to train gpflow model in batches. Trained model will be checkpointed and saved.
    """
    tf_optimization_step = tf.function(optimization_step)

    batches = iter(train_dataset)
    loss = list()
    for epoch in range(epochs):
        step_loss = 0
        for step in range(ci_niter(num_batches_per_epoch)):
            step_loss += tf_optimization_step(
                model,
                next(batches),
                optimizer=optimizer,
                natgrad_opt=natgrad_opt,
            )
            if step_var is not None:
                step_var.assign(epoch * num_batches_per_epoch + step + 1)
        loss.append(step_loss.numpy())
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        # monitor(epoch, epoch_id=epoch_id, data=data)
        if epoch_id % logging_epoch_freq == 0:
            ckpt_path = manager.save()
            tf.print(
                f"Epoch {epoch_id}: LOSS (train):{loss[epoch]} , saved at {ckpt_path}"
            )
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()
    plt.savefig('loss_plot_{}.png'.format(exp_tag), dpi=100)


def mean_squared_error(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def checkpointing_training_loop_exact_GP(
    model: gpflow.models.GPR,
    X: tf.Tensor,
    Y: tf.Tensor,
    epochs: int,
    manager: tf.train.CheckpointManager,
    optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
    logging_epoch_freq: int = 10,
    epoch_var: Optional[tf.Variable] = None,
    exp_tag: str = 'test',
):
    tf_optimization_step = tf.function(optimization_exact)
    
    loss = list()
    for epoch in range(epochs):
        tf_optimization_step(model)
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        loss.append(model.training_loss())
        if epoch_id % logging_epoch_freq == 0:
            ckpt_path = manager.save()
            tf.print(
                f"Epoch {epoch_id}: LOSS (train) {model.training_loss()}, saved at {ckpt_path}"
            )
            tf.print(f"MSE: {mean_squared_error(Y, model.predict_y(X)[0])}")
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()
    plt.show()
#     plt.savefig('{}/loss_plot_{}.png'.format(exp_tag, exp_tag), dpi=100)
    
def only_training_loop_exact_GP(
    model: gpflow.models.GPR,
    X: tf.Tensor,
    Y: tf.Tensor,
    epochs: int,
    optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01),
    logging_epoch_freq: int = 10,
    epoch_var: Optional[tf.Variable] = None,
):
    tf_optimization_step = tf.function(optimization_exact)
    
    loss = list()
    for epoch in range(epochs):
        tf_optimization_step(model)
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        loss.append(model.training_loss())
        if epoch_id % logging_epoch_freq == 0:
            tf.print(
                f"Epoch {epoch_id}: LOSS (train) {model.training_loss()}, MSE {mean_squared_error(Y, model.predict_y(X)[0])}")
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()
