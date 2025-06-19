from core.environment import setup_environment
from config.argument_parser import parse_args
from training.trainer import ModelTrainer
from config.model_config import ModelConfig

def main():
  """Main execution function."""
  # Parse arguments and setup environment
  setup_environment()
  config: ModelConfig = parse_args()
  
  # Initialize and run training
  trainer = ModelTrainer(config)
  model, history = trainer.train()
  
if __name__ == '__main__':
  main()