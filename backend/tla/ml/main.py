import sys
from train import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_dataset>")
        sys.exit(1)

    data_path = sys.argv[1]
    main(data_path)
