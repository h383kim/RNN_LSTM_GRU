import torch
from torch import nn

class LSTMCell(nn.Module):
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


    # X_in, h_prev, c_prev are the inputs to LSTM Cell
    def forward(self, x, h_prev, c_prev):
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
        