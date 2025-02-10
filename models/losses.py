import tensorflow as tf

def quaternion_loss(y_true, y_pred):
    """
    Computes the loss between true and predicted quaternions.

    Args:
        y_true: Tensor of shape (batch_size, 4) representing true quaternions.
        y_pred: Tensor of shape (batch_size, 4) representing predicted quaternions.

    Returns:
        Tensor representing the mean loss over the batch.
    """
    # Calculate the squared inner product between true and predicted quaternions
    inner_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    loss = 1.0 - tf.square(inner_product)
    return tf.reduce_mean(loss)