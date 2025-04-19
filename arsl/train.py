import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from arsl.baseline_model import BaselineModel
from arsl.data import get_dataloaders
from arsl.utils import get_device, load_checkpoint, save_checkpoint


def train(model, loss_fn, data, optimizer, device):
    running_loss = 0.0  # Running loss for intermediate printing
    total_loss = 0.0  # Accumulate total loss for the entire epoch
    correct_predictions = 0
    total_predictions = 0

    model.train()

    for batch_index, (inputs, targets) in enumerate(data):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass: compute model output
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and total loss
        running_loss += loss.item()
        total_loss += loss.item()

        _, predicted = outputs.max(1)
        correct_predictions += predicted.eq(targets).sum().item()
        total_predictions += targets.size(0)

        # Print loss every 10 batches
        if batch_index % 10 == 0 and batch_index > 0:
            print(f"The running Loss: {running_loss:.4f}")
            running_loss = 0.0

    # Calculate average loss and accuracy for the entire epoch
    avg_loss = total_loss / len(data)
    accuracy = (100.0 * correct_predictions / total_predictions)
    
    print(f"End of Epoch: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Accumulate the loss
            val_loss += loss.item()

            # Calculate the accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate average loss and accuracy
    avg_val_loss = val_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_val_loss, accuracy


def main(args):
    device = get_device()
    train_loader, test_loader = get_dataloaders(
        root_dir=args.root_dir, batch_size=args.batch_size
    )

    model = BaselineModel(
        args.conv_size,
        args.stride,
        args.lstm_input,
        args.hidden_size,
        args.num_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    epochs_res = [
        os.path.join(args.checkpoints_dir, res)
        for res in sorted(os.listdir(args.checkpoints_dir))
    ]

    if epochs_res:
        start_epoch = load_checkpoint(model, optimizer, epochs_res[-1]) + 1
        print(f"[INFO] Loaded checkpoint from {epochs_res[-1]}")

    else:
        start_epoch = 0
        print("[INFO] No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        
        print(f"We are in {epoch + 1}th epoch")
        
        avg_train_loss, training_acc = train(
            model,
            loss_fn,
            data=train_loader,
            optimizer=optimizer,
            device=device
            )

        print(f"The training loss is {avg_train_loss}, and the Training Accuracy is {training_acc}")
        
        print(f"Now we are testing the Model after {epoch + 1}th epoch")

        avg_val_loss, accuracy = test(model, test_loader, loss_fn, device)
        
        print(f"The Average loss is: {avg_val_loss} and the accuracy is {accuracy}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, args.checkpoints_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Sign Language Recognition Model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr", 
        type=float,
        default=1e-3,
        help="Learning rate"
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for dataloaders"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "-cs",
        "--conv_size",
        nargs="*",
        type = int,
        required=True,
        help="List CNN sizes"
    )

    parser.add_argument(
        "-s",
        "--stride",
        type = int,
        required=True,
        nargs="*",
        help="List that represent the stride for each CNN layer",
    )

    parser.add_argument(
        "-li",
        "--lstm_input",
        required=True,
        type=int,
        help="The input size for the lstm",
    )

    parser.add_argument(
        "-hs",
        "--hidden_size",
        required=True,
        type=int,
        help="The hidden size for the lstm",
    )

    parser.add_argument(
        "-nl",
        "--num_layers",
        required=True,
        type=int,
        help="The number of lstm layers"
    )

    args = parser.parse_args()

    main(args)
