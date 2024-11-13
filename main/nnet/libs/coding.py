import torch as th
import os
from trainer import SiSnrTrainer, Trainer  # Make sure to import the trainer classes
from node import MS_SL2_split_model  # Adjust import as per your setup
from dataset import make_dataloader  # Make sure DataLoader is properly implemented

def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_model(checkpoint_dir, data_loader, device):
    """Evaluate the model for each saved checkpoint."""
    model = MS_SL2_split_model()  # Initialize with proper arguments if needed
    model = model.to(device)
    trainer = SiSnrTrainer(model, loss_mode="snr")  # Initialize your trainer

    for epoch, filename in enumerate(sorted(os.listdir(checkpoint_dir))):
        if filename.endswith(".pt.tar"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            load_checkpoint(model, checkpoint_path, device)
            model.eval()
            
            total_loss = 0.0
            total_batches = 0
            with th.no_grad():
                for egs in data_loader:
                    loss = trainer.compute_loss(egs)  # Ensure this method is correctly implemented
                    total_loss += loss.item()
                    total_batches += 1
            avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
            print(f"Epoch {epoch + 1}: Checkpoint {filename} - Loss = {avg_loss}")

def main():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    data_config = {
        "mix_scp": "/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/cv/mix.scp",  # adjust paths as necessary
        "ref_scp": ["/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/cv/spk1.scp", "/media/speech70809/Data01/speech_donoiser_new/datasets/ner-300hr/cv/spk2.scp"],
        "sample_rate": 16000
    }
    data_loader = make_dataloader(train=False, data_kwargs=data_config, batch_size=16, num_workers=4)
    # data_loader = make_dataloader(train=False, data_kwargs=data_config, batch_size=1, num_workers=4)

    checkpoint_dir = '/media/speech70809/Data01/speech_donoiser_new/model_2024_03_05_Fusion'
    evaluate_model(checkpoint_dir, data_loader, device)

if __name__ == "__main__":
    main()
