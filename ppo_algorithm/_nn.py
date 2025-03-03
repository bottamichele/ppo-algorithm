from torch.nn import Module, Sequential, Linear, ReLU
from torch.optim import Adam

class NN(Module):
    """A fully connected neural network for Actor or Critic."""

    def __init__(self, input_dim, output_dim, fc_dim, fc_num):
        """Create new neural network to Actor or Critic.

        Parameters:
        --------------------
        input_dim: int
            input dimension

        output_dim: int
            output dimension

        fc_dim: int
            number of nodes for each hidden layer

        fc_num: int
            number of hidden layers to use
        """
        
        super().__init__()

        #Check the parameters.
        if input_dim <= 0:
            raise ValueError("input_dim must be an integer positive.")
        
        if output_dim <= 0:
            raise ValueError("output_dim must be an integer positive.")

        if fc_dim <= 0:
            raise ValueError("fc_dim must be an integer positive.")
        
        if fc_num <= 0:
            raise ValueError("fc_num must be an integer positive.")

        #Hidden layers.
        self._hidden = Sequential()

        for i in range(fc_num):
            if i == 0:
                #First hidden layer.
                self._hidden.append(Linear(input_dim, fc_dim))
            else:
                #i-th hidden layer.
                self._hidden.append(Linear(fc_dim, fc_dim))
            self._hidden.append(ReLU())

        #Output layer.
        self._out = Linear(fc_dim, output_dim)

    def forward(self, x):
        """Compute x.
        
        Parameter
        --------------------
        x: torch.Tensor
            a tensor
            
        Return
        --------------------
        y: torch.Tensor
            x computed by this neural network"""
        
        x = self._hidden(x)
        return self._out(x)