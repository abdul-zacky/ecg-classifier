# Clinical ECG Classification with Multi-Branch Deep Learning

A comprehensive deep learning project for ECG (Electrocardiogram) classification using a multi-branch neural network architecture. This project implements state-of-the-art techniques for clinical ECG interpretation with real-time analysis capabilities.

## Project Overview

This project presents a novel multi-branch deep learning approach for clinical ECG classification that combines:
- **Rhythm Analysis Branch**: 1D ResNet for temporal rhythm pattern detection
- **Morphology Analysis Branch**: 2D CNN for local morphological feature extraction  
- **Global Interpretation Branch**: 2D CNN for comprehensive ECG pattern analysis

The model achieves **94.24% macro AUC** and **75.73% F-max score** on the PTB-XL dataset, demonstrating state-of-the-art performance in automated ECG interpretation.

## Key Features

- **Multi-branch Architecture**: Three specialized neural network branches for comprehensive ECG analysis
- **Clinical Evidence Integration**: Real-time clinical criteria evaluation with medical interpretation
- **71 Cardiac Conditions**: Classification across a comprehensive range of cardiac abnormalities
- **Interactive Visualization**: Clinical-grade ECG visualization with automated annotation
- **Real-time Analysis**: Fast inference suitable for clinical deployment

## Supported Cardiac Conditions

The model can detect and classify 71 different cardiac conditions including:

### Rhythm Abnormalities
- Atrial Fibrillation (AFIB)
- Atrial Flutter (AFL)
- Supraventricular Tachycardia (SVTAC)
- Sinus Tachycardia (STACH)
- Sinus Bradycardia (SBRAD)
- Ventricular Tachycardia (VTAC)

### Conduction Abnormalities
- AV Blocks (1st, 2nd, 3rd degree)
- Bundle Branch Blocks (LBBB, RBBB)
- Fascicular Blocks (LAFB, LPFB)

### Myocardial Infarction
- Anterior MI (AMI)
- Inferior MI (IMI)
- Lateral MI (LMI)
- Anterolateral MI (ALMI)

### Hypertrophy & Structural Changes
- Left Ventricular Hypertrophy (LVH)
- Right Ventricular Hypertrophy (RVH)
- Atrial Enlargement (LAE, RAE)

*[Complete list of 71 conditions available in the clinical documentation]*

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `torch>=2.0.0`: Deep learning framework
- `torchvision`: Computer vision utilities
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.20.0`: Numerical computing
- `scikit-learn>=1.0.0`: Machine learning utilities
- `matplotlib>=3.5.0`: Visualization
- `seaborn>=0.11.0`: Statistical visualization
- `wfdb`: ECG signal processing
- `scipy>=1.8.0`: Scientific computing

## Project Structure

```
├── ClinicalECG_Notebook.ipynb    # Complete Jupyter notebook implementation
├── ClinicalECG_Python.py         # Standalone Python script
├── requirements.txt              # Project dependencies
├── ptbxl_database.csv           # PTB-XL dataset metadata
├── sample_prediction.png        # Example ECG analysis output
└── src/
    ├── main.py                  # Main training/inference script
    ├── models.py                # Multi-branch model architecture
    ├── train.py                 # Training utilities and loops
    ├── utils.py                 # Data loading and preprocessing
    ├── config.py                # Configuration parameters
    └── intrep_clinical.py       # Clinical interpretation engine
```

## Quick Start

### Option 1: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook ClinicalECG_Notebook.ipynb
```

### Option 2: Using Python Scripts
```bash
# Train the model
python src/main.py

# Run inference on new ECG data
python src/intrep_clinical.py --input ecg_file.npy
```

### Option 3: Standalone Script
```bash
python ClinicalECG_Python.py
```

## Model Architecture

### Multi-Branch Design
```
ECG Signal (12 leads × 1000 samples) → Input Layer
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
  Rhythm Branch  Morphology   Global Branch
   (1D ResNet)   Branch       (2D CNN)
                 (2D CNN)
        ↓           ↓           ↓
    512 features 512 features 512 features
        └───────────┼───────────┘
                    ↓
            Fusion Layer (1536 features)
                    ↓
            Classification Head
                    ↓
          71 Cardiac Conditions
```

### Branch Specifications
- **Rhythm Branch**: 1D ResNet with 4 layers, specialized for temporal pattern recognition
- **Morphology Branch**: 2D CNN with attention mechanisms for morphological analysis
- **Global Branch**: 2D CNN for comprehensive multi-lead pattern integration

## Training Process

### Dataset
- **PTB-XL Dataset**: 21,837 ECG recordings
- **Sampling Rate**: 100 Hz
- **Duration**: 10 seconds per recording
- **Labels**: 71 cardiac conditions (multi-label classification)

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 1e-3 with ReduceLROnPlateau
- **Optimizer**: AdamW with weight decay
- **Loss Function**: BCEWithLogitsLoss
- **Epochs**: 25 (with early stopping)

### Data Preprocessing
- Bandpass filtering (0.5-40 Hz)
- Z-score normalization per lead
- Multi-label binarization
- Stratified train/val/test split (80%/10%/10%)

## Performance Metrics

### Test Set Results
- **Macro AUC**: 94.24%
- **F-max Score**: 75.73%
- **Optimal Threshold**: 0.3
- **Valid AUC Labels**: 71/71

### Branch Contribution Analysis
- **Rhythm Branch**: Average activation magnitude 6.07
- **Morphology Branch**: Average activation magnitude 8.84 (highest)
- **Global Branch**: Average activation magnitude 5.71

## Clinical Integration

### Clinical Criteria Engine
The system includes a sophisticated clinical interpretation engine that evaluates:
- Heart rate and rhythm analysis
- QRS width and morphology
- ST-segment changes (elevation/depression)
- T-wave abnormalities
- P-wave analysis
- Axis deviation assessment

### Real-time Visualization
- Clinical-grade ECG grid overlay
- Automated annotation of key findings
- Color-coded abnormality highlighting
- Comprehensive diagnostic summary

## Usage Examples

### Basic ECG Classification
```python
from src.models import ClinicalECGClassifier
from src.utils import PTBXLDataset
import torch

# Load model
model = ClinicalECGClassifier(num_classes=71)
model.load_state_dict(torch.load('path/to/model.pth'))

# Classify ECG
ecg_signal = torch.randn(1, 12, 1000)  # Batch size 1, 12 leads, 1000 samples
predictions = torch.sigmoid(model(ecg_signal))
```

### Clinical Analysis
```python
from src.intrep_clinical import ECGClinicalAnalyzer

analyzer = ECGClinicalAnalyzer()
clinical_results = analyzer.evaluate_clinical_criteria(ecg_signal, 'AFIB')
print(f"Confidence: {clinical_results['confidence']:.2%}")
```

## Model Interpretability

### Feature Visualization
- Branch-specific feature activation maps
- Clinical criteria correlation analysis
- Attention visualization for key ECG segments
- Lead-specific contribution analysis

### Clinical Validation
- Automated clinical criteria checking
- Medical guideline compliance verification
- Confidence scoring based on evidence strength
- Differential diagnosis ranking

## Deployment Options

### Local Deployment
```bash
# Start local inference server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t ecg-classifier .

# Run container
docker run -p 8000:8000 ecg-classifier
```

### Cloud Deployment
Compatible with:
- AWS SageMaker
- Google Cloud AI Platform
- Azure Machine Learning

## Research & Citations

This project implements and extends techniques from:
- Multi-branch neural architectures for ECG analysis
- Clinical ECG interpretation guidelines
- Deep learning for cardiac arrhythmia detection

### Dataset Citation
```bibtex
@article{wagner2020ptb,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and others},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={1--15},
  year={2020}
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone repository
git clone https://github.com/abdul-zacky/kecerdasan-dibuat-buat.git
cd kecerdasan-dibuat-buat

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Medical Disclaimer

**IMPORTANT**: This software is intended for research and educational purposes only. It is NOT approved for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## Support & Contact

- **Issues**: [GitHub Issues](https://github.com/abdul-zacky/kecerdasan-dibuat-buat/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abdul-zacky/kecerdasan-dibuat-buat/discussions)
- **Email**: abdul.zacky@ui.ac.id muttaqin.muzakkir@gmail.com irma.nia@ui.ac.id

## Acknowledgments

- PTB-XL dataset contributors
- PyTorch community
- Clinical cardiology experts who provided domain knowledge
- Open source contributors

---

**Made with ❤️ for advancing cardiac healthcare through AI**