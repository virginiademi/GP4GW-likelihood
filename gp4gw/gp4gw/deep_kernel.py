import gpflow
import tensorflow as tf
from typing import Optional

class DeepKernel(gpflow.kernels.Kernel):

    def __init__(
        self,
	data_dim: int,
        output_dim: int,
        base_kernel: gpflow.kernels.Kernel,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        with self.name_scope:
            self.base_kernel = base_kernel
            self.feature_extractor = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(128, input_dim=data_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1), dtype=tf.float64),
                tf.keras.layers.Dense(128, input_dim=data_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1), dtype=tf.float64),
                tf.keras.layers.Dense(128, input_dim=data_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1), dtype=tf.float64),
                    tf.keras.layers.Dense(output_dim, dtype=tf.float64)
                                          ])
            self.feature_extractor.build()

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.feature_extractor(a_input)
        transformed_b = self.feature_extractor(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.feature_extractor(a_input)
        return self.base_kernel.K_diag(transformed_a)
