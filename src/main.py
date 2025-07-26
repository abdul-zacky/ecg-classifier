# main.py
import torch
from config import DATASET_PATH, NPY_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
from utils import PTBXLDataset, train_df, val_df, test_df, all_labels, train_loader, val_loader, test_loader
from models import ClinicalECGClassifier
from train import ECGTrainer, evaluate_model
from intrep_clinical import ECGClinicalAnalyzer, analyze_ecg_with_clinical_evidence

def main():
    # Initialize device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Initialize model
    model = ClinicalECGClassifier(num_classes=len(all_labels))
    model = model.to(device)
    print(f"Model initialized with {len(all_labels)} classes")
    
    # Initialize trainer
    trainer = ECGTrainer(model, train_loader, val_loader, all_labels, device)
    
    # Train the model
    print("\nStarting training...")
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, all_labels, device)
    
    # Initialize clinical analyzer
    analyzer = ECGClinicalAnalyzer()
    
    # Example clinical analysis
    print("\nclinical interpretation")
    test_sample = test_df[1967] 
    sample_signal = test_sample['signal']
    
    clinical_results = analyze_ecg_with_clinical_evidence(
        model, sample_signal, all_labels, device, analyzer, threshold=0.5
    )

if __name__ == "__main__":
    main()