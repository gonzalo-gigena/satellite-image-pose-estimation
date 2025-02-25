from core.environment import setup_environment
from config.argument_parser import parse_args
from train.trainer import ModelTrainer
from evaluation.metrics import analyze_history

def main():
  """Main execution function."""
  # Parse arguments and setup environment
  args = parse_args()
  setup_environment()
  
  # Initialize and run training
  trainer = ModelTrainer(args)
  model, history = trainer.train()

  analyze_history(history)
  
  # Save model
  trainer.save_model()

if __name__ == '__main__':
  main()