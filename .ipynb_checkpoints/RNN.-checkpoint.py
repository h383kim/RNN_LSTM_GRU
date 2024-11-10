import torch
from torch import nn

class RNNCell(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.cell = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.Tanh()
        )

    def forward(self, x):
        return self.cell(x)


class RNN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=1):
        super().__init__()
        self.hidden_ch = hidden_ch   # Channel size the hidden cell outputs
        self.num_layers = num_layers # Number of cells layered in one token

        self.cells = nn.ModuleList([
            RNNCell(in_ch if i == 0 else self.hidden_ch, hidden_ch) 
            for i in range(self.num_layers)
        ])

        self.W_t = nn.Linear(self.hidden_ch, out_ch)
        
    def forward(self, x, hidden=None):
        '''
        Forward pass for the entire sequence

        Args:
            x: Input sequence, (batch_size, context_length, embedding_dim)
        '''
        bs, seq_len, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(self.num_layers, bs, self.hidden_ch, device=x.device, dtype=x.dtype)

        y_t = [] # List of outputs at each token of sequence
        h_t = hidden
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t[layer] = self.cells[layer](x_t if layer == 0 else h_t[layer - 1])
            # h_t[-1]: [batch_size, hidden_ch]
            # output: [batch_size, out_ch]
            output = self.W_t(h_t[-1])
            y_t.append(output)

        output_final = torch.stack(y_t, dim=1)

        return output_final, h_t