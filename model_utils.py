import torch
import requests
from io import BytesIO
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):   
    for epoch in range(epochs):
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {train_loss/len(train_loader):.3f}.. "
            f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
            f"Validation accuracy: {accuracy/len(valid_loader)*100:.2f}%")
        
        model.train()
    return model

def build_model():
   
    # URL for the VGG16 model weights
    vgg16_weights_url = "https://download.pytorch.org/models/vgg16-397923af.pth"

    # Download the weights without SSL verification
    response = requests.get(vgg16_weights_url, verify=False)
    weights_data = BytesIO(response.content)
    # Load a pre-trained model
    model = models.vgg16()
    model.load_state_dict(torch.load(weights_data))

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a custom classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),  # Input size depends on the selected model
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),  # 102 output classes for flowers dataset
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

def save_checkpoint(model, train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
    'arch': 'vgg16',
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, 'checkpoint.pth')

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array.
    """
    # Define the transformations for image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Open and preprocess the image
    image = Image.open(image_path)
    image = preprocess(image)
    
    # Convert to Numpy array
    np_image = image.numpy()
    
    return np_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Load and preprocess the image
    image = process_image(image_path)
    
    # Convert the Numpy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image)
    
    # Add a batch dimension and move tensor to the GPU if available
    image_tensor = image_tensor.unsqueeze(0).float()
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # Perform the forward pass through the model
    with torch.no_grad():
        output = model(image_tensor)
    
    # Calculate class probabilities
    probabilities = torch.exp(output)
    
    # Get the top K classes and their indices
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert tensors to lists
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Invert class_to_idx to get idx_to_class
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Map indices to class labels
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probabilities, top_classes