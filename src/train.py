from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
class ECGTrainer:
    def __init__(self, model, train_loader, val_loader, all_labels, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_f1s = []
        self.all_labels = all_labels
        self.device = device

        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.best_model_path = '/content/drive/MyDrive/datathon/models/ecg_model.pth'

        # Tensorboard
        self.writer = SummaryWriter('logs/ecg_multibranch')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training')

        for batch_idx, batch in enumerate(pbar):
            signals = batch['signal'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/Train_Step', loss.item(),
                                     epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')

            for batch in pbar:
                signals = batch['signal'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(signals)
                loss = self.criterion(logits, labels)

                probs = torch.sigmoid(logits)
                
                all_predictions.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        aucs = []
        for i in range(len(self.all_labels)):
            try:
                if len(np.unique(all_labels[:, i])) > 1:
                    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                    aucs.append(auc)
            except:
                continue

        macro_auc = np.mean(aucs) if aucs else 0.0

        thresholds = np.arange(0.1, 0.9, 0.1)
        f_scores = []
        for threshold in thresholds:
            pred_binary = (all_predictions > threshold).astype(int)
            f_score = f1_score(all_labels, pred_binary, average='samples', zero_division=0)
            f_scores.append(f_score)

        f_max = max(f_scores) if f_scores else 0.0

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        self.val_aucs.append(macro_auc)
        self.val_f1s.append(f_max)

        return avg_loss, macro_auc, f_max

    def train(self, num_epochs=50, resume=False):
        print(f"Starting training for {num_epochs} epochs...")

        start_epoch = 0
        if resume and os.path.exists(self.best_model_path):
            start_epoch, self.best_auc, self.train_losses, self.val_losses, self.val_aucs, self.val_f1s = load_checkpoint(
                self.model, self.optimizer, self.scheduler, self.best_model_path, self.device
            )


        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*60)

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_auc, val_fmax = self.validate(epoch)
            self.writer.add_scalar('F1/Val', val_fmax, epoch)

            # Scheduler step
            self.scheduler.step(val_auc)

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('AUC/Val', val_auc, epoch)
            self.writer.add_scalar('F-max/Val', val_fmax, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f}")
            print(f"Val F-max: {val_fmax:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")


            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_auc': self.best_auc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'all_labels': self.all_labels,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_aucs': self.val_aucs,
                    'val_f1s': self.val_f1s
                }, self.best_model_path)
                print(f"New best model saved! AUC: {val_auc:.4f}")

        self.writer.close()
        print(f"\n Training completed! Best AUC: {self.best_auc:.4f}")
        return self.best_auc

def evaluate_model(model, test_loader, all_labels, device):
    model.eval()

    all_predictions = []
    all_labels_true = []
    all_branch_features = {'rhythm': [], 'morphology': [], 'global': []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            signals = batch['signal'].to(device)
            labels = batch['labels'].to(device)

            # Get predictions
            logits = model(signals)
            probs = torch.sigmoid(logits)

            # Get branch features
            branch_features = model.get_branch_features(signals)

            # Store results
            all_predictions.append(probs.cpu().numpy())
            all_labels_true.append(labels.cpu().numpy())

            for branch_name, features in branch_features.items():
                all_branch_features[branch_name].append(features.cpu().numpy())

    # Combine results
    predictions = np.vstack(all_predictions)
    labels_true = np.vstack(all_labels_true)

    for branch_name in all_branch_features:
        all_branch_features[branch_name] = np.vstack(all_branch_features[branch_name])

    # Calculate metrics
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)

    # Overall metrics
    aucs = []
    for i in range(len(all_labels)):
        try:
            if len(np.unique(labels_true[:, i])) > 1:
                auc = roc_auc_score(labels_true[:, i], predictions[:, i])
                aucs.append(auc)
        except:
            continue

    macro_auc = np.mean(aucs) if aucs else 0.0

    # F-max
    thresholds = np.arange(0.1, 0.9, 0.1)
    f_scores = []
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        f_score = f1_score(labels_true, pred_binary, average='samples', zero_division=0)
        f_scores.append(f_score)

    f_max = max(f_scores) if f_scores else 0.0
    best_threshold = thresholds[np.argmax(f_scores)]

    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"F-max: {f_max:.4f} (threshold: {best_threshold:.1f})")
    print(f"Number of labels with valid AUC: {len(aucs)}/{len(all_labels)}")

    return {
        'predictions': predictions,
        'labels_true': labels_true,
        'branch_features': all_branch_features,
        'macro_auc': macro_auc,
        'f_max': f_max,
        'best_threshold': best_threshold,
        'individual_aucs': aucs
    }

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    val_aucs = checkpoint.get('val_aucs', [])
    val_f1s = checkpoint.get('val_f1s', [])

    return checkpoint['epoch'], checkpoint['best_auc'], train_losses, val_losses, val_aucs, val_f1s

# Initialize trainer
print("Initializing ECG Trainer...")
trainer = ECGTrainer(model, train_loader, val_loader, all_labels, device)

resume = False

# Start training
print("Starting ECG Multi-Branch Model Training...")
print(f"- Labels: {len(all_labels)} classes")
print(f"- Training samples: {len(train_dataset)}")
print(f"- Validation samples: {len(val_dataset)}")