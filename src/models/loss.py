import tensorflow as tf
import numpy as np

def quaternion_loss(y_true, y_pred):
    """
    Computes the loss between true and predicted quaternions.

    Args:
        y_true: Tensor of shape (batch_size, 4) representing true quaternions.
        y_pred: Tensor of shape (batch_size, 4) representing predicted quaternions.

    Returns:
        Tensor representing the mean loss over the batch.
    """
    # Calculate the squared inner product between true and predicted quaternions (already normalized)
    inner_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    loss = 1.0 - tf.square(inner_product)
    return tf.reduce_mean(loss)

def angular_distance_loss(y_true, y_pred):
    """
    Computes the angular distance between true and predicted quaternions.
    
    Args:
        y_true: Tensor of shape (batch_size, 4) representing true quaternions
        y_pred: Tensor of shape (batch_size, 4) representing predicted quaternions
        
    Returns:
        Tensor representing the mean angular distance over the batch (in radians)
    """
    
    # Calculate dot product
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    
    # Clamp dot product to [-1, 1] to avoid numerical issues
    dot_product = tf.clip_by_value(tf.abs(dot_product), 0.0, 1.0)
    
    # Calculate angular distance (multiply by 2 for actual angle)
    angle = 2.0 * tf.acos(dot_product)
    
    return tf.reduce_mean(angle)

def detailed_distance_loss(y_true, y_pred):
    """
    Computes the minimum distance between quaternions considering both q and -q.
    
    Args:
        y_true: Tensor of shape (batch_size, 4) representing true quaternions
        y_pred: Tensor of shape (batch_size, 4) representing predicted quaternions
        
    Returns:
        Tensor representing the mean minimum distance over the batch
    """
    
    # Calculate distance to q
    dist_positive = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
    
    # Calculate distance to -q
    dist_negative = tf.sqrt(tf.reduce_sum(tf.square(y_true + y_pred), axis=-1))
    
    # Take the minimum of the two distances
    min_distance = tf.minimum(dist_positive, dist_negative)
    
    return tf.reduce_mean(min_distance)

def geodesic_loss(y_true, y_pred):
    # Assuming both y_true and y_pred are already normalized to unit quaternions
    # Compute the dot product for each quaternion pair along the last axis
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    # Use absolute value to account for double covering (q and -q represent the same rotation)
    dot = tf.abs(dot)
    # Clip dot values to avoid numerical issues with acos
    dot = tf.clip_by_value(dot, -1.0, 1.0)
    # Compute the angular difference (in radians)
    angle = 2 * tf.acos(dot)
    # Return the mean angular difference over the batch
    return tf.reduce_mean(angle)