import os
import torch # For the model
import torch.nn as nn # For the neural network
from torchvision import datasets, transforms # For the datasets and transformations
from torch.utils.data import DataLoader # For the data loader
import timm # For the ResNet50 model
import time # For the time measurement
from collections import Counter # For the counter
from collections import OrderedDict # For the ordered dictionary

#list with the categories of the tomato dataset
categories = [
    "Saludable", "Virus_Mosaico", "Virus_rizado_amarillo", "Mancha_Diana", "Acaro_Araña_Roja_Dos_Puntos",
    "Mancha_Foliar_Por_Septoria", "Moho_Foliar", "Tizon_Tardio", "Tizon_Temprano", "Mancha_Bacteriana"
            ]

def image_transforms():
    # 1. Define the transformations for the images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.ToTensor(), # Convert the images to tensors (values between 0 and 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) # Normalize the images
    ])

    # The same transformations are used for the validation and test datasets
    # The only difference is that the RandomHorizontalFlip transformation is not used because it is not necessary
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val

def load_datasets(train_path, val_path, test_path, transform_train, transform_val):
    # 2. Load the datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    class_names = train_dataset.classes

    # Extraer todas las etiquetas del dataset
    labels = [sample[1] for sample in train_dataset.samples]
    
    # Contar la frecuencia de cada etiqueta
    counts = Counter(labels)
    
    # Mostrar el resultado con el nombre de cada clase
    print("\n -------------------INFORMACIÓN DEL DATASET ----------------------\n")
    class_names = train_dataset.classes
    for label, count in counts.items():
        print(f"{class_names[label]}: {count} imágenes")
        
    # save the list of categories in a file
    with open("Resources/categories.txt", "w") as file:
        for category in class_names:
            file.write(category + "\n")

    print("--------------------------------------------------------------------\n")
    
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform_val)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform_val)
    print("Datasets loaded")

    # set the train, validation and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    print("Data loaders set")
    return train_loader, val_loader, test_loader

def create_model():
    # 3. Define the ResNet50 model
    model = timm.create_model('resnet50', pretrained=True)
    print("Model defined")

    # 4. Modify the last layer of the model to have 10 output classes (one for each category)
    model.fc = nn.Linear(model.fc.in_features, len(categories))
    print("Last layer modified")

    return model

def train_model(model, train_loader, val_loader, epochs=10):
    
    # 5. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Loss function and optimizer defined")
    
    # 6. Train the model
    model, device = set_model_device(model)
    print(f"Training on {device}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
    print("--------------------------------------------------------------------\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs} \n")
        i = 0
        for images, labels in train_loader:
            i += 1
            actual_time = time.time()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            time_spend_in_sec = time.time()-actual_time
            print(f"Batch #{i} Process time {round(time_spend_in_sec, 2)}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Loss: {epoch_loss:.4f}")

        # 7. Validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        print("Validating...")
        with torch.no_grad():
            i = 0
            for images, labels in val_loader:
                i += 1
                actual_time = time.time()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
                time_spend_in_sec = time.time()-actual_time

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

    print("Training completed.")
    return model

def save_model(model, path = "Resources\\TomatoResNet50.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def test_model_with_loader(model, test_loader, criterion):
    model, device = set_model_device(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
     # 8. Test the model
    model.eval()
    test_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct.double() / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def test_model_with_image(model, image, transform_val):
    model, device = set_model_device(model)
    model.eval()
    with torch.no_grad():
        image = transform_val(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return preds.item()
    
def load_model(path = "Resources\\TomatoResNet50.pth"):
    model = create_model()
    model, device = set_model_device(model)
    state_dict = torch.load(path, map_location=device)
    if str(device) == "cpu" or (device == "cuda" and torch.cuda.device_count() == 1):
        print("removing module.")
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            # Si la clave empieza con "module.", la recortamos
            if key.startswith("module."):
                new_key = key[len("module."):]
            else:
                new_key = key
            new_state_dict[new_key] = val


    model.load_state_dict(new_state_dict)
    return model

def set_model_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    return model, device

def get_categories(path = "Resources\\categories.txt"):
    with open(path, "r", encoding= "utf-8") as file:
        categories = file.read().split("\n")
    return categories

#if  __name__ == "__main__":
#   
#    train_path = "../Resources/Tomato-Dataset/Train/"
#    val_path = "../Resources/Tomato-Dataset/Validation/"
#    test_path = "../Resources/Tomato-Dataset/Test/"
#    
#    transform_train, transform_val = image_transforms()
#    train_loader, val_loader, test_loader = load_datasets(train_path, val_path, test_path, transform_train, transform_val)
#    
#    # only test the model
#    model = load_model()
    