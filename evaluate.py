import argparse
import torch
from dataloader import get_data_loader
from adacof import AdaCoFNet
import os
import sys

def evaluate(args):
    # Load test data
    try:
        test_loader = get_data_loader(args.test_data_path, args.batch_size, shuffle=False)
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        sys.exit(1)
    
    # Load model
    model = AdaCoFNet(kernel_size=5).to(args.device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        sys.exit(1)
    
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            
            # Calculate metrics (assuming helper functions for PSNR and SSIM)
            psnr = compute_psnr(outputs, targets)
            ssim = compute_ssim(outputs, targets)
            total_psnr += psnr
            total_ssim += ssim
            count += 1

    print(f"Average PSNR: {total_psnr / count:.2f}, Average SSIM: {total_ssim / count:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AdaCoFNet model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation")
    args = parser.parse_args()

    evaluate(args)
