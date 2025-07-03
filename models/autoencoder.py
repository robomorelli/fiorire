import torch
from torch import nn
import math
import os

class AE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.original_dim = kwargs["original_dim"]
        self.intermediate_dim = kwargs["intermediate_dim"]
        self.code_dim = kwargs["code_dim"]

        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["original_dim"], out_features=kwargs['intermediate_dim']
        )
        self.encoder_output_layer = nn.Linear(
            in_features=kwargs['intermediate_dim'], out_features=kwargs['code_dim']
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs['code_dim'], out_features=kwargs['intermediate_dim']
        )
        self.decoder_output_layer = nn.Linear(
            in_features=kwargs['intermediate_dim'], out_features=kwargs["original_dim"]
        )

        if 'checkpoint_dir' in kwargs.keys():
            self.checkpoint_dir = kwargs['checkpoint_dir']
        else:
            self.checkpoint_dir = None

        if self.checkpoint_dir is not None:
            model_state, optimizer_state = torch.load(self.checkpoint_dir)
            self.load_state_dict(model_state)
            # self.optimizer.load_state_dict(optimizer_state)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            print(module)
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)


    def forward(self, x):
        activation = self.encoder_hidden_layer(x.view(x.size(0), self.original_dim))
        activation = nn.ReLU(True)(activation)
        code = self.encoder_output_layer(activation)
        code = nn.ReLU(True)(code)
        activation = self.decoder_hidden_layer(code.view(code.size(0), self.code_dim))
        activation = nn.ReLU(True)(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.nn.Sigmoid()(activation)
        return reconstructed
