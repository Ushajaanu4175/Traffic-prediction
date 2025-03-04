import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lstm_layers=2, dropout=0.5):
        super(TemporalCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # ✅ Add Batch Normalization after first convolution
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # ✅ Add Batch Normalization after second convolution
        
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=lstm_layers, 
                            batch_first=True, dropout=dropout, bidirectional=False)
        
        self.fc = nn.Linear(hidden_dim, 1)  # Output single value per sequence

    def forward(self, x):
        # ✅ Ensure x has 3 dimensions
        if x.dim() == 2:  
            x = x.unsqueeze(1)  # Convert (batch_size, input_dim) -> (batch_size, 1, input_dim)
        
        x = x.permute(0, 2, 1)  # ✅ Fix: Transpose to (batch_size, input_dim, sequence_length)
        
        # Convolutional Layers with Batch Normalization
        x = self.conv1(x)
        x = self.bn1(x)  # ✅ Normalize
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)  # ✅ Normalize
        x = self.relu(x)
        
        x = x.permute(0, 2, 1)  # ✅ Convert back to (batch_size, sequence_length, input_dim)
        
        _, (hn, _) = self.lstm(x)  # Get last hidden state
        
        return self.fc(hn[-1])  # Fully connected layer output

