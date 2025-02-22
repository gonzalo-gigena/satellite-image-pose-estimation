from models.loss import quaternion_loss, angular_distance_loss, detailed_distance_loss

def get_loss_function(loss_name):
  """Select and return the loss function based on the given name"""
  loss_functions = {
    'quaternion': quaternion_loss,
    'angular': angular_distance_loss,
    'detailed': detailed_distance_loss,
  }

  loss_function = loss_functions.get(loss_name.lower())
  if loss_function is None:
    raise ValueError(f"Unsupported loss function: {loss_function}")

  return loss_function