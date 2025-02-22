from .args import parse_arguments
from .filter import delete_empty_images

def main():
  args = parse_arguments()
  delete_empty_images(args.path, args.threshold, args.delete)

if __name__ == "__main__":
  main()