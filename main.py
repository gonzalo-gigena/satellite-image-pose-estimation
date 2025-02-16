import tensorflow as tf
  
from history_analysis import analyze_model_performance
from utils import get_optimizer, get_data_loader, get_loss_function, get_model, get_train_generator

def main(args):
  """Main function to execute training based on provided arguments.

  Args:
      args (argparse.Namespace): Parsed command-line arguments.
  """
  # Get the appropriate model
  model = get_model(args.matching_method)
  
  # Get the appropriate data loader
  data_loader = get_data_loader(
    data_path=args.data_path,
    train_split=args.train_split,
    validation_split=args.validation_split,
    seed=args.seed,
    matching_method=args.matching_method,
    num_matches=args.num_matches
  )
  
  # Load data
  data = data_loader.load_data()
  
  # Create data generators
  train_generator = get_train_generator(
    data['train'],
    args.batch_size,
    args.matching_method,
    shuffle=True
  )
  
  val_generator = get_train_generator(
    data['val'],
    args.batch_size,
    args.matching_method,
    shuffle=False
  )      

  # Get the optimizer
  optimizer = get_optimizer(args.optimizer, args.lr)

  # Get loss function
  loss_function = get_loss_function(args.loss)

  # Train the model
  model, history = train_model(
    model=model,
    train_generator=train_generator,
    val_generator=val_generator,
    optimizer=optimizer,
    loss=loss_function,
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
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss',
          factor=0.5,
          patience=5,
          min_lr=1e-6
      ),
      tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=15,
          restore_best_weights=True
      )
  ]

  # Train the model
  history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    #callbacks=callbacks,
  )

  return model, history