import argparse
from train import main as train_main

def main():
    """
    Main entry point for the SuperCombo training program
    """
    parser = argparse.ArgumentParser(description='SuperCombo Neural Network Training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, required=True, 
                            help='Directory containing dataset')
    train_parser.add_argument('--dataset', type=str, default='cityscapes', 
                            choices=['cityscapes', 'comma10k'], 
                            help='Dataset to use for training')
    train_parser.add_argument('--batch_size', type=int, default=8, 
                            help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, 
                            help='Number of epochs')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, 
                            help='Learning rate')
    train_parser.add_argument('--limit_samples', type=int, default=None, 
                            help='Limit number of samples (for debugging)')
    train_parser.add_argument('--annotation_mode', type=str, default='fine', 
                            choices=['fine', 'coarse'], 
                            help='Annotation mode for Cityscapes dataset')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == 'train':
        train_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
