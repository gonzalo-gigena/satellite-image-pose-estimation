from config.argument_parser import parse_args
from config.model_config import ModelConfig
from core.environment import setup_environment
from training.trainer import ModelTrainer


def main():
  """Main execution function."""
  # Parse arguments and setup environment
  config: ModelConfig = parse_args()
  setup_environment()

  # Initialize and run training
  trainer = ModelTrainer(config)
  model, history = trainer.train()


if __name__ == '__main__':
  main()
