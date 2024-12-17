import lightning as L
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from model import SimpleTibetanNumberClassifier

def main(data_path: str):
    # Load Dataset
    dataset = CustomDataset(data_path)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize the model
    model = SimpleTibetanNumberClassifier(num_classes=10)

    # Initialize the Trainer
    trainer = L.Trainer(max_epochs=10, gpus=1)  # Set gpus=0 if you are not using a GPU

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

if __name__ == "__main__":
    data_path = "TibetanMNIST/Datasets/TibetanMNIST.npz"  # Replace with your actual data path in npz file format
    main(data_path)
