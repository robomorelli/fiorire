from models.vae import VAE
from models.autoencoder import AE
from models.lstm_ae import LSTM_AE
from models.lstm import LSTM
from models.conv_ae import CONV_AE
from models.conv_ae1D import CONV_AE1D
from models.lstm_vae import LSTM_VAE
from models.cnn3d import CNN3D

def get_model(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :return: model
    """
    if cfg.model.name == "vae":
        model = VAE(cfg, original_dim=kwargs['original_dim'], intermediate_dim=kwargs['intermediate_dim'],
                    latent_dim=kwargs['latent_dim'],
                    n_lognorm=kwargs['n_lognorm'], n_binomial=kwargs['n_binomial'])
        return model

    elif cfg.model.name == "ae":
        model = AE(cfg, original_dim=kwargs['original_dim'], intermediate_dim=kwargs['intermediate_dim'],
                    code_dim=kwargs['code_dim'])
        return model

    elif cfg.model.name == "lstm_ae":
        model = LSTM_AE(cfg, **kwargs)
        print('loading lstm_ae')
        print(model)
        return model

    elif cfg.model.name == "lstm":
        model = LSTM(cfg, **kwargs)
        print('loading lstm')
        print(model)
        return model

    elif cfg.model.name == "conv_ae":
        model = CONV_AE(cfg, **kwargs)
        print('loading conv_ae')
        print(model)
        return model

    elif cfg.model.name == "conv_ae1D":
        model = CONV_AE1D(cfg, **kwargs)
        print('loading conv_ae 1D')
        print(model)
        return model

    elif cfg.model.name == "lstm_vae":
        model = LSTM_VAE(cfg, **kwargs)
        print('loading lstm_vae')
        print(model)
        return model

    elif cfg.model.name == "cnn3d":
        model = CNN3D(cfg, **kwargs)
        return model
