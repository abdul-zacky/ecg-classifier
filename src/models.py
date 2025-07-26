import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import all_labels

class ResNet1DBlock(nn.Module):
    """1D ResNet block for rhythm analysis"""
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RhythmBranch(nn.Module):
    """1D ResNet for rhythm analysis"""
    def __init__(self, in_channels=12, base_filters=64):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7)
        self.initial_bn = nn.BatchNorm1d(base_filters)
        self.initial_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_size = base_filters * 8

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNet1DBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x

class MorphologyBranch(nn.Module):
    """2D CNN for local morphology analysis"""
    def __init__(self, base_filters=64):
        super().__init__()

        # 2D convolutions to analyze lead patterns
        self.conv1 = nn.Conv2d(1, base_filters, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))

        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))

        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))

        self.conv4 = nn.Conv2d(base_filters*4, base_filters*8, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(base_filters*8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_size = base_filters * 8

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class GlobalBranch(nn.Module):
    """2D CNN for global ECG interpretation"""
    def __init__(self, base_filters=64):
        super().__init__()

        self.conv1 = nn.Conv2d(1, base_filters, kernel_size=(6, 15), stride=(1, 4), padding=(2, 7))
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(1, 2), padding=(0, 1))

        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=(4, 11), stride=(1, 3), padding=(1, 5))
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=(1, 2), padding=(0, 1))

        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))

        self.conv4 = nn.Conv2d(base_filters*4, base_filters*8, kernel_size=(2, 5), stride=(1, 2), padding=(0, 2))
        self.bn4 = nn.BatchNorm2d(base_filters*8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_size = base_filters * 8

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x

class ClinicalECGClassifier(nn.Module):
    """Multi-branch ECG model for comprehensive analysis"""
    def __init__(self, num_classes=71, base_filters=64):
        super().__init__()

        # Three branches
        self.rhythm_branch = RhythmBranch(base_filters=base_filters)
        self.morphology_branch = MorphologyBranch(base_filters=base_filters)
        self.global_branch = GlobalBranch(base_filters=base_filters)

        # Combined feature size
        total_features = (self.rhythm_branch.output_size +
                         self.morphology_branch.output_size +
                         self.global_branch.output_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        rhythm_features = self.rhythm_branch(x)

        x_2d = x.unsqueeze(1)

        morphology_features = self.morphology_branch(x_2d)
        global_features = self.global_branch(x_2d)

        combined_features = torch.cat([
            rhythm_features,
            morphology_features,
            global_features
        ], dim=1)

        logits = self.classifier(combined_features)

        return logits

    def get_branch_features(self, x):
        batch_size = x.size(0)

        rhythm_features = self.rhythm_branch(x)
        x_2d = x.unsqueeze(1)
        morphology_features = self.morphology_branch(x_2d)
        global_features = self.global_branch(x_2d)

        return {
            'rhythm': rhythm_features,
            'morphology': morphology_features,
            'global': global_features
        }

# Test model
model = ClinicalECGClassifier(num_classes=len(all_labels))
model = model.to(device)

# Test forward pass
test_input = torch.randn(4, 12, 1000).to(device)
with torch.no_grad():
    output = model(test_input)
    branch_features = model.get_branch_features(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Rhythm features: {branch_features['rhythm'].shape}")
print(f"Morphology features: {branch_features['morphology'].shape}")
print(f"Global features: {branch_features['global'].shape}")