# src/run.py
from core.environment import setup_environment
from config.argument_parser import parse_args
from train.trainer import ModelTrainer
from evaluation.metrics import analyze_model_performance

def main():
  """Main execution function."""
  # Parse arguments and setup environment
  args = parse_args()
  setup_environment()
  
  # Initialize and run training
  trainer = ModelTrainer(args)
  model, history = trainer.train()
  
  # Analyze performance
  metrics = analyze_model_performance(
    model=model,
    history=history,
    test_data={
      'image_data': trainer.data['train']['image_data'],
      'numerical': trainer.data['train']['numerical']
    },
    test_targets=trainer.data['train']['targets']
  )
  
  # Print results
  print("Training complete. Performance metrics:")
  for key, value in metrics.items():
    print(f"{key}: {value}")
  
  # Save model
  trainer.save_model()

if __name__ == '__main__':
  main()