import torch as tc

from torch.nn import Module, Sequential, Conv2d, Linear, ReLU

def action_size_conv2d(input_dim, kernel_size, stride, padding):
    """Return the output dimension (width and height) of a 2D convolution layer used in a CNN.
    
    Parameters
    --------------------
    input_dim: tuple
        input dimension of a 2D convolution layer
        
    kernel_size: int
        kernel size of 2D convolution layer
        
    stride: int
        stride of a 2D convolution layer
        
    padding: int
        padding of a 2D convolution layer"""
    
    assert input_dim[0] > 0 and input_dim[1] > 0, "Input dimension of a 2D convolution layer must be only positive integers."

    w = input_dim[0]
    h = input_dim[1]
    return (w + 2 * padding - kernel_size) // stride + 1, (h + 2 * padding - kernel_size) // stride + 1,


class CnnActorCritic(Module):
    """A 2D convolutional neural network of Actor and Critic."""

    def __init__(self, action_size, obs_size=(4, 84, 84), normalize_input=True, conv_layers=[(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)], fc_dim=512, fc_num=1, actor_fc_num=1, critic_fc_num=1):
        """Create new 2D convolutional neural network of Actor and Critic.
        
        Parameters:
        --------------------
        action_size: int
            action size
            
        obs_size: tuple, optional
            observation size. The tuple must be as (ch, h, w) where 
            w is the image width, h is image height and ch is number of channels.

        normalize_input: bool, optional
            whethere or not to normalize inputs between 0.0 and 1.0
            
        conv_layers: list, optional
            list of 2D convolutional layer parameters. The list must contains tuples as (n_ch, k, s, p) where
            n_ch is number of channels, k is kernel size, s is stride and p is padding.
            
        fc_dim: int, optional
            number of nodes for a fully connected layer
            
        fc_num: int, optional
            number of fully connected layers
            
        actor_fc_num: int
            number of Actor's hidden layers

        critic_fc_num: int
            number of Critic's hidden layers"""
        
        super().__init__()

        if action_size <= 0:
            raise ValueError("action_size isn't a positive integer.")

        if len(conv_layers) <= 0:
            raise ValueError("conv_layers must contain at least one 2D convolutional layer parameter.")
        
        if fc_dim <= 0:
            raise ValueError("fc_dim isn't a positive integer.")

        if fc_num <= 0:
            raise ValueError("fc_num isn't a positive integer.")
        
        if actor_fc_num <= 0:
            raise ValueError("actor_fc_num must be an integer positive.")

        if critic_fc_num <= 0:
            raise ValueError("critic_fc_num must be an integer positive.")

        self._normalize_input = normalize_input

        #2D convolution layers.
        self._conv_layers = Sequential()
        for i in range(len(conv_layers)):
            n_channels, kernel_size, stride, padding = conv_layers[i]

            if i == 0:
                #First 2D convolution layer.
                self._conv_layers.append(Conv2d(obs_size[0], n_channels, kernel_size, stride, padding))
            else:
                #i-th 2D convolution layer.
                self._conv_layers.append(Conv2d(conv_layers[i - 1][0], n_channels, kernel_size, stride, padding))
        
            self._conv_layers.append(ReLU())

        #Fully connected layers.
        self._fc_layers = Sequential()
        for i in range(fc_num):
            if i == 0:
                #First fully connected layer.
                _, height, width = obs_size
                for (_, k, s, p) in conv_layers:
                    width, height = action_size_conv2d((width, height), k, s, p)
                
                self._fc_layers.append(Linear(conv_layers[-1][0] * width * height, fc_dim))
            else:
                #i-th fully connected layer.
                self._fc_layers.append(Linear(fc_dim, fc_dim))

            self._fc_layers.append(ReLU())

        #Actor's hidden layers.
        self._actor_layers = Sequential()

        for i in range(actor_fc_num):
            if i == actor_fc_num - 1:
                #Output layer.
                self._actor_layers.append(Linear(fc_dim, action_size))
            else:
                #i-th hidden layer.
                self._actor_layers.append(Linear(fc_dim, fc_dim))
                self._actor_layers.append(ReLU())

        #Critic's hidden layers.
        self._critic_layers = Sequential()

        for i in range(critic_fc_num):
            if i == critic_fc_num - 1:
                #Output layer.
                self._critic_layers.append(Linear(fc_dim, 1))
            else:
                #i-th hidden layer.
                self._critic_layers.append(Linear(fc_dim, fc_dim))
                self._critic_layers.append(ReLU())

    def forward(self, x):
        """Process x.
        
        Parameter
        --------------------
        x: tc.Tensor
            a tensor
            
        Return
        --------------------
        act_value: torch.Tensor
            action values
            
        value_state: torch.Tensor
            value-state value"""
        
        x = x.to(dtype=tc.float32)
        if self._normalize_input:
            x /= 255.0

        x = self._conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self._fc_layers(x)
        return self._actor_layers(x), self._critic_layers(x)