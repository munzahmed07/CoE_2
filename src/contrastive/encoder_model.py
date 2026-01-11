# src/contrastive/encoder_model.py
import torch
import torch.nn as nn

# After
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_size=3, embedding_size=64, hidden_size=128):
        super().__init__()
        # LSTM layer to process the sequence of (x, y) points
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        # A simple linear layer to create the final embedding
        self.projection_head = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        # Input x shape: [batch_size, sequence_length, 2]
        # We only need the final hidden state from the LSTM
        _, (hidden_state, _) = self.lstm(x)
        
        # Take the hidden state of the last layer
        last_hidden = hidden_state[-1] # Shape: [batch_size, hidden_size]
        
        # Project it to the final embedding size
        embedding = self.projection_head(last_hidden)
        return embedding