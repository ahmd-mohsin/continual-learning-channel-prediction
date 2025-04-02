# import argparse
# import os
# from model import CustomLSTMModel, load_model
# from dataloader import ChannelSequenceDataset
# from utils import train_model, evaluate_model, compute_device
# import torch

# def main():
#     parser = argparse.ArgumentParser(description="Train an LSTM model for channel prediction.")
#     parser.add_argument("--ext", type=str, required=True, choices=["npy", "mat"], help="Dataset file extension (npy or mat)")
#     parser.add_argument("--file_path", type=str, default="../dataset/outputs/umi_compact_conf_2tx_2rx.", help="Dataset file path without extension")
#     args = parser.parse_args()

#     torch.manual_seed(42)
    
#     device = compute_device()
#     file_path = args.file_path
#     dataset_file = file_path + '.' + args.ext  # Add extension to the file path
#     full_dataset = ChannelSequenceDataset(file_path, args.ext, device)
    
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
    
#     train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
#     batch_size = 16
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
#     print(f"Created dataloaders with batch size {batch_size}")

#     # Initialize model and move to device
#     model = CustomLSTMModel().to(device)

#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params}")

#     # Create a folder based on the file path to store log and model files
#     output_dir = os.path.join(os.path.basename(file_path))
#     os.makedirs(output_dir, exist_ok=True)

#     # Define the log file and model save paths
#     training_log_file = os.path.join(output_dir, "training_log.csv")
#     print(f"Training log file: {training_log_file}")
#     evaluation_log_file = os.path.join(output_dir, "evaluation_log.csv")
#     model_save_path = os.path.join(output_dir, "best_channel_predictor.pth")

#     print(f"Starting training...")
#     model = train_model(model=model, dataloader=train_dataloader, device=device, num_epochs=30, learning_rate=1e-3, log_file=training_log_file, model_save_path=model_save_path)
    
#     print("Evaluating model...")
#     val_loss = evaluate_model(model, val_dataloader, device, log_file=evaluation_log_file)
    
#     print("Training completed!")
#     print(f"Final validation loss: {val_loss:.6f}")

# if __name__ == "__main__":
#     main()


import argparse
import os
import torch

# Import the new model classes from model.py:
from model import (
    MLPModel,
    CNNModel,
    GRUModel,
    LSTMModel,
    TransformerModel
)
# If you still want to use load_model for inference, you can also import it:
# from model import load_model

from dataloader import ChannelSequenceDataset
from utils import train_model, evaluate_model, compute_device


def main():
    parser = argparse.ArgumentParser(description="Train a channel-prediction model.")
    parser.add_argument("--ext", type=str, required=True, choices=["npy", "mat"],
                        help="Dataset file extension (npy or mat)")
    parser.add_argument("--file_path", type=str, default="../dataset/outputs/umi_compact_conf_2tx_2rx.",
                        help="Dataset file path without extension")
    # New argument to choose model architecture:
    parser.add_argument("--model_type", type=str, default="LSTM",
                        choices=["MLP", "CNN", "GRU", "LSTM", "TRANS"],
                        help="Which model architecture to train.")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(42)
    
    device = compute_device()
    file_path = args.file_path
    
    # Load dataset
    full_dataset = ChannelSequenceDataset(file_path, args.ext, device)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Created dataloaders with batch size {batch_size}")

    # ----------------------------------------------------------------------------
    # Instantiate the desired model. Adjust dimensions as needed:
    # (Below we assume your channel has size H=18, W=8, and input seq_len=16, etc.)
    # ----------------------------------------------------------------------------
    
    if args.model_type == "MLP":
        model = MLPModel(
            input_dim=16 * 2 * 18 * 8,  # example if your seq_len=16, and 2*18*8 for real+imag
            hidden_dim=128,
            H=18,
            W=8
        ).to(device)

    elif args.model_type == "CNN":
        model = CNNModel(
            in_channels=2,    # "2" used per time-slice, though we group seq_len inside
            H=18,
            W=8,
            seq_len=16,       # adjust if your overlapping_index=16
            hidden_channels=32
        ).to(device)

    elif args.model_type == "GRU":
        model = GRUModel(
            input_dim=1,      # not strictly used—since we flatten to 2*H*W
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=18,
            W=8
        ).to(device)

    elif args.model_type == "LSTM":
        model = LSTMModel(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            n_layers=3,
            H=18,
            W=8
        ).to(device)

    elif args.model_type == "TRANS":
        model = TransformerModel(
                dim_val=128,
                n_heads=4,
                n_encoder_layers=1,
                n_decoder_layers=1,
                out_channels=4,  # Because dataloader outputs (4,18,2)
                H=18,
                W=2,
                seq_len=16
            ).to(device)
   

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using {args.model_type} model. Total parameters: {total_params}")

    # Create output folder
    output_dir = os.path.basename(file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Define log file & model save paths
    training_log_file = os.path.join(output_dir, "training_log.csv")
    evaluation_log_file = os.path.join(output_dir, "evaluation_log.csv")
    model_save_path = os.path.join(output_dir, "best_channel_predictor.pth")
    
    print(f"Training log file: {training_log_file}")
    print(f"Starting training...")

    # Train
    model = train_model(
        model=model,
        dataloader=train_dataloader,
        device=device,
        num_epochs=30,
        learning_rate=1e-3,
        log_file=training_log_file,
        model_save_path=model_save_path
    )
    
    # Evaluate
    print("Evaluating model...")
    val_loss = evaluate_model(model, val_dataloader, device, log_file=evaluation_log_file)
    
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
