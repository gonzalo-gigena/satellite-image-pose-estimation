import tensorflow as tf

from data_loader import MatchingDataLoader, GrayscaleDataLoader
from models.feature_matching_model import (
  DataGenerator as MatchingDataGenerator,
  FeatureMatchingModel
)
from models.grayscale_model import (
  DataGenerator as GrayscaleDataGenerator,
  GrayscaleModel,
)
from models.losses import quaternion_loss  # Import the shared loss function
from history_analysis import analyze_model_performance

def get_optimizer(optimizer_name, learning_rate):
  """Select and return the optimizer based on the given name."""
  optimizers = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
    'rmsprop': tf.keras.optimizers.RMSprop,
  }
  optimizer_class = optimizers.get(optimizer_name.lower())
  if optimizer_class is None:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
  return optimizer_class(learning_rate=learning_rate)

def train_model(
  model,
  train_generator,
  val_generator,
  optimizer,
  loss,
  epochs,
):
  """Train the given model with the provided data generators."""
  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['mae'],
  )

  # Define callbacks
  callbacks = [
    tf.keras.callbacks.EarlyStopping(
      patience=10,
      restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
      factor=0.5,
      patience=5,
    ),
  ]

  # Train the model
  history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    #callbacks=callbacks,
  )

  return model, history

def main(args):
  """Main function to execute training based on provided arguments.

  Args:
      args (argparse.Namespace): Parsed command-line arguments.
  """
  # Select appropriate data loader and model based on matching_method
  if args.matching_method is not None:
    # Using feature matching data
    data_loader = MatchingDataLoader(
      data_path=args.data_path,
      train_split=args.train_split,
      validation_split=args.validation_split,
      seed=args.seed,
      matching_method=args.matching_method,
      num_matches=args.num_matches,
    )

    # Load data
    data = data_loader.load_data()

    # Create data generators
    train_generator = MatchingDataGenerator(
      points=data['train']['image_data'],
      numerical=data['train']['numerical'],
      targets=data['train']['targets'],
      batch_size=args.batch_size,
    )

    val_generator = MatchingDataGenerator(
      points=data['val']['image_data'],
      numerical=data['val']['numerical'],
      targets=data['val']['targets'],
      batch_size=args.batch_size,
    )

    # Instantiate the model
    model = FeatureMatchingModel()

  else:
    # Using grayscale images
    data_loader = GrayscaleDataLoader(
      data_path=args.data_path,
      train_split=args.train_split,
      validation_split=args.validation_split,
      seed=args.seed,
    )

    # Load data
    data = data_loader.load_data()

    # Create data generators
    train_generator = GrayscaleDataGenerator(
      images=data['train']['image_data'],
      numerical=data['train']['numerical'],
      targets=data['train']['targets'],
      batch_size=args.batch_size,
    )

    val_generator = GrayscaleDataGenerator(
      images=data['val']['image_data'],
      numerical=data['val']['numerical'],
      targets=data['val']['targets'],
      batch_size=args.batch_size,
    )

    # Instantiate the model
    model = GrayscaleModel()

  # Get the optimizer
  optimizer = get_optimizer(args.optimizer, args.learning_rate)

  # Train the model
  model, history = train_model(
    model=model,
    train_generator=train_generator,
    val_generator=val_generator,
    optimizer=optimizer,
    loss=quaternion_loss,
    epochs=args.epochs,
  )

  # Analyze and print performance metrics
  metrics = analyze_model_performance(
    model=model,
    history=history,
    test_data={
      'image_data': data['train']['image_data'],
      'numerical': data['train']['numerical']
    },
    test_targets=data['train']['targets'],
  )

  print("Training complete. Performance metrics:")
  for key, value in metrics.items():
    print(f"{key}: {value}")

  # Optionally, save the trained model
  if args.model_save_path:
    model.save(args.model_save_path)
    print(f"Model saved to {args.model_save_path}")