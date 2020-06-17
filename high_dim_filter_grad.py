import tensorflow as tf
from tensorflow.python.framework import ops
custom_module = tf.load_op_library('./cpp/high_dim_filter.so')


@ops.RegisterGradient("HighDimFilter")
def _high_dim_filter_grad(op, grad):

    rgb = op.inputs[1]
    grad_vals = custom_module.high_dim_filter(grad, rgb,
                                              bilateral=op.get_attr("bilateral"),
                                              theta_alpha=op.get_attr("theta_alpha"),
                                              theta_beta=op.get_attr("theta_beta"),
                                              theta_gamma=op.get_attr("theta_gamma"),
                                              backwards=True)

    return [grad_vals, tf.zeros_like(rgb)]
