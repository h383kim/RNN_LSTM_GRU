import torch
from torch import nn

class LSTMCell(nn.Module):
    """
    A custom implementation of an LSTM cell.

    Args:
        input_dim (int): The number of expected features in the input `x`.
        hidden_dim (int): The number of features in the hidden state `h`.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        ''' 
        forget_gate f_t:
            f_t = sigmoid(W_forget*[h_prev, X_in] + b_forget
        input_gate i_t:
            i_t = sigmoid(W_input*[h_prev, X_in] + b_input) * tanh(W_cell*[h_prev, X_in] + b_cell)
        next cell_state c_t:
            c_t = f_t*c_prev + i_t
        output_gate o_t:
            o_t = sigmoid(W_output*[h_prev, X_in] + b_output)
        next hidden_state h_t:
            h_t = o_t + tanh(c_t)
        '''
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)


    # Inputs to LSTM Cell are X_in, h_prev, c_prev
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single time step of the LSTM cell.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.
            h_prev (Tensor): Previous hidden state of shape `(batch_size, hidden_dim)`.
            c_prev (Tensor): Previous cell state of shape `(batch_size, hidden_dim)`.

        Returns:
            h_t (Tensor): Current hidden state of shape `(batch_size, hidden_dim)`.
            c_t (Tensor): Current cell state of shape `(batch_size, hidden_dim)`.
        """
        
        # Concatenate input and previous hidden state [h_prev, X_in]
        combined = torch.cat([x, h_prev], axis=1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined)) * torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        # Next cell_state c_t
        c_t = f_t*c_prev + i_t
        # Next hidden_state h_t
        h_t = o_t + torch.tanh(c_t)

        # Return new hidden_state and new cell_state
        return h_t, c_t





# LSTM Module
class LSTM(nn.Module):
    """
    A custom implementation of a multi-layer LSTM network.

    Args:
        input_dim (int): The number of expected features in the input `x`.
        hidden_dim (int): The number of features in the hidden state `h`.
        output_dim (int, optional): The number of features in the output. If None, the output size will be `hidden_size`.
        num_layers (int, optional): Number of recurrent layers. Default is 1.

    Attributes:
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        cells (nn.ModuleList): List of LSTMCell instances for each layer.
        output_layer (nn.Linear or None): Linear layer to map hidden state to output size.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            self.lstm_cells.append(
                # If num_layers > 0, second layer and onwards will take hidden_dim as input dimension
                LSTMCell(input_dim if i==0 else hidden_dim, hidden_dim)
            )
            
        # Output layer to map hidden state to desired output size
        self.output_layer = nn.Linear(hidden_dim, output_dim) if output_dim else None


    
    def forward(self, x, hidden=None):
        """
        Forward pass for the LSTM network.

        Args:
            x (Tensor): Input tensor of shape `(seq_len, batch_size, input_size)`.
            hidden (tuple, optional): A tuple `(h_0, c_0)` containing the initial hidden state and cell state for each layer.
                - h_0: Tensor of shape `(num_layers, batch_size, hidden_size)`.
                - c_0: Tensor of shape `(num_layers, batch_size, hidden_size)`.

        Returns:
            outputs (Tensor): Output tensor of shape `(seq_len, batch_size, output_size or hidden_size)`.
            (h_n, c_n) (tuple): Tuple containing the final hidden state and cell state for each layer.
                - h_n: List of tensors, each of shape `(batch_size, hidden_size)`.
                - c_n: List of tensors, each of shape `(batch_size, hidden_size)`.
        """
        
        batch_size, seq_len = x.size()
        # If hidden_states and cell_states are not given, initialize
        if not hidden:
            c_t = [torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype) 
                   for _ in range(self.num_layers)]
                      
            h_t = [torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
                   for _ in range(self.num_layers)]
        else:
            h_t, c_t = hidden

        # Initialize y_t
        outputs = []
        # Sequentially feed forward the input using for loop
        for t in range(seq_len):
            x_t = x[:, t, :] # Slicing the input at timestep t
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_cells[layer](x_t, h_t[layer], c_t[layer])
                x_t = h_t[layer] # in a multi-layer LSTM, input for second layer and onwards are just the hidden state from above layer
            # y_t is the last hidden_state feed forwarded a fully connected layer if exists
            if self.output_layer:
                y_t = self.output_layer(h_t[-1])
            else:
                y_t = h_t[-1]
            # Append to the outputs
            outputs.append(y_t.squeeze(0))

        outputs = torch.cat(outputs, dim=0)
        
        return outputs, (h_t, c_t)