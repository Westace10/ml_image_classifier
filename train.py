import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from model_utils import build_model, train_model, save_checkpoint

# Define the main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a neural network to classify flowers")

    # Add arguments
    parser.add_argument("data_directory", type=str, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Architecture (vgg16 or densenet121)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    # Parse the arguments
    args = parser.parse_args()

    # Define data transformations
    data_transforms = {
        "train": transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "valid": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # Load the data
    image_datasets = {x: datasets.ImageFolder(f"{args.data_directory}/{x}", transform=data_transforms[x]) for x in ["train", "valid", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ["train", "valid", "test"]}

    # Build the model
    model = build_model(args.arch, args.hidden_units)

    # Train the model
    criterion = nn.NLLLoss()

    # Use Adam 
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Use GPU if available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Move the model to the device
    model.to(device)

    # Train the model
    train_model(model, dataloaders["train"], dataloaders["valid"], criterion, optimizer, device, args.epochs)

    # Save the model checkpoint
    save_checkpoint(model, image_datasets["train"], args.save_dir)

if __name__ == "__main__":
    main()
