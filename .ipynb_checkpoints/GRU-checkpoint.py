# GRU Cell Implementation
class GRUCell(nn.Module):
    """
    A custom implementation of a GRU cell.

    Args:
        input_dim (int): The number of expected features in the input `x`.
        hidden_dim (int): The number of features in the hidden state `h`.

    Attributes:
        hidden_dim (int): The number of features in the hidden state.
        reset_gate (nn.Linear): Linear layer for the reset gate.
        update_gate (nn.Linear): Linear layer for the update gate.
        new_gate (nn.Linear): Linear layer for the candidate hidden state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Gates: reset, update, new
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_size, hidden_dim)
        self.new_gate = nn.Linear(input_dim + hidden_size, hidden_dim)
    
    def forward(self, x, h_prev):
        """
        Forward pass for a single time step of the GRU cell.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.
            h_prev (Tensor): Previous hidden state of shape `(batch_size, hidden_dim)`.

        Returns:
            h_t (Tensor): Current hidden state of shape `(batch_size, hidden_dim)`.
        """
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        
        # Reset gate
        r_t = torch.sigmoid(self.reset_gate(combined))
        # Update gate
        z_t = torch.sigmoid(self.update_gate(combined))
        # New gate (candidate hidden state)
        combined_new = torch.cat((x, r_t * h_prev), dim=1)
        n_t = torch.tanh(self.new_gate(combined_new))
        
        # New hidden state
        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t

# GRU Module Implementation with Output Layer
class GRU(nn.Module):
    """
    A custom implementation of a multi-layer GRU network.

    Args:
        input_dim (int): The number of expected features in the input `x`.
        hidden_dim (int): The number of features in the hidden state `h`.
        output_dim (int, optional): The number of features in the output. If None, the output dim will be `hidden_dim`.
        num_layers (int, optional): Number of recurrent layers. Default is 1.

    Attributes:
        hidden_dim (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        cells (nn.ModuleList): List of GRUCell instances for each layer.
        output_layer (nn.Linear or None): Linear layer to map hidden state to output dim.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(
                GRUCell(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim
                )
            )
        
        # Output layer to map hidden state to desired output dim
        self.output_layer = nn.Linear(hidden_dim, output_dim) if output_dim else None
    
    def forward(self, x, hidden=None):
        """
        Forward pass for the GRU network.

        Args:
            x (Tensor): Input tensor of shape `(seq_len, batch_size, input_dim)`.
            hidden (Tensor, optional): Initial hidden state for each layer of shape `(num_layers, batch_size, hidden_dim)`.

        Returns:
            outputs (Tensor): Output tensor of shape `(seq_len, batch_size, output_dim or hidden_dim)`.
            h_n (list): List containing the final hidden state for each layer, each of shape `(batch_size, hidden_dim)`.
        """
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            h_t = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_t = [hidden[i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                h_t[layer] = self.cells[layer](x_t, h_t[layer])
                x_t = h_t[layer]  # Input for next layer
            # Apply output layer if defined
            if self.output_layer:
                out_t = self.output_layer(h_t[-1])
            else:
                out_t = h_t[-1]
            outputs.append(out_t.unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)
        return outputs, h_t