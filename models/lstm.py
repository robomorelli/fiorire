import torch.nn as nn
import torch

torch.manual_seed(0)


class ENCODER(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1, n_cells=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(ENCODER, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_cells = n_cells

        if self.n_cells == 1:
            # define LSTM layer
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
            self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def init_hidden(self, batch_size):

        # initialize hidden state
        #: param batch_size:    x_input.shape[1]
        #: return:              zeroed hidden state and cell state

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        if self.n_cells == 1:
            lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        else:
            lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
            lstm_out, _ = self.lstm1(lstm_out)

        return lstm_out, self.hidden


class LSTM(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        self.hidden_size = model_cfg.embedding_dim
        self.n_layers = model_cfg.n_layers
        self.n_cells = model_cfg.n_cells

        self.seq_in = cfg.dataset.seq_in_length
        self.seq_out = cfg.dataset.seq_out_length  # Add in traine
        self.input_size = cfg.dataset.n_features

        self.encoder = ENCODER(self.input_size,
                                    self.hidden_size, self.n_layers, self.n_cells)

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):

        # outputs tensor
        outputs = torch.zeros(x.shape[0], self.seq_out, self.input_size)
        # initialize hidden state
        # encoder_output, encoder_hidden = self.encoder(x)

        # decoder_input = x[:, -1, :].unsqueeze(1)  # shape: (batch_size, input_size)
        # decoder_hidden = encoder_hidden

        encoder_hidden = (
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size, requires_grad=True).to(x.device),
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size, requires_grad=True).to(x.device))

        for t in range(self.seq_out):
            # decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            encoder_output, encoder_hidden = self.encoder(x, (encoder_hidden))
            one_step_forecast = self.linear(encoder_hidden[0][-1:, :, :])
            one_step_forecast = torch.permute(one_step_forecast, (1, 0, 2))
            x = torch.cat([x[:, 1:, :], one_step_forecast], axis=1)
            outputs[:, t, :] = one_step_forecast.squeeze()

            # outputs[:,t,:] = decoder_output.squeeze(1)
            # decoder_input = decoder_output

        return outputs

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))

    ''' 
        def predict(self, input_tensor, target_len):
    
    
            #: param input_tensor:      input data (seq_len, input_size); PyTorch tensor
            #: param target_len:        number of target values to predict
            #: return np_outputs:       np.array containing predicted values; prediction done recursively
    
    
            # encode input_tensor
            input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
            encoder_output, encoder_hidden = self.encoder(input_tensor)
    
            # initialize tensor for predictions
            outputs = torch.zeros(target_len, input_tensor.shape[2])
    
            # decode input_tensor
            decoder_input = input_tensor[-1, :, :]
            decoder_hidden = encoder_hidden
    
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output.squeeze(0)
                decoder_input = decoder_output
    
            np_outputs = outputs.detach().numpy()
    
            return
    '''