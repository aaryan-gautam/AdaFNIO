import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_data_loader
from adacof import AdaCoFNet  # Assuming adacof.py contains AdaCoFNet
import torch.optim as optim
import torch.nn as nn
import os
import sys

def train(args):
    # Set up data loader
    try:
        train_loader = get_data_loader(args.train_data_path, args.batch_size)
        val_loader = get_data_loader(args.val_data_path, args.batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        sys.exit(1)
    
    # Initialize model, criterion, optimizer, and learning rate scheduler
    model = AdaCoFNet(kernel_size=5).to(args.device)
    criterion = nn.MSELoss()  # Replace with an appropriate loss function if needed
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # TensorBoard setup for logging
    writer = SummaryWriter(log_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % args.log_interval == 0:
                print(f"[{epoch + 1}, {i}] loss: {running_loss / args.log_interval:.3f}")
                writer.add_scalar("Loss/train", running_loss / args.log_interval, epoch * len(train_loader) + i)
                running_loss = 0.0

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

        scheduler.step()  # Adjust learning rate
        model.train()

    print("Training completed.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AdaCoFNet model")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()

    train(args)
