import argparse
import torch
import json
from torchvision import models, transforms
from model_utils import load_checkpoint, process_image, predict


def main():
    # Parse the command line arguments 
    parser = argparse.ArgumentParser(description="Predict the flower name from an image")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Mapping of categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load the model checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    # Process the input image
    processed_image = process_image(args.input)

    # Predict the class probabilities
    probabilities, classes = predict(processed_image, model, class_to_idx, args.top_k)

    # Print the top K classes and their probabilities
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    flower_names = [cat_to_name[cls] for cls in classes]

    print("Top K Classes:")
    for i in range(args.top_k):
        print(f"{flower_names[i]}: {probabilities[i]:.2f}")

if __name__ == "__main__":
    main()
