import torch
from models.utils.layers import clip_x_to0


def KL_loss(mu, log_var):
    #kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).sum(-1))
    #kl = kl.sum(-1)     # to go from multi-dimensional z to single dimensional z : (batch_size x latent_size) ---> (batch_size)
                            # i.e Z = [ [z1_1, z1_2 , ...., z1_lt] ] ------> z = [ z1]
                            #         [ [z2_1, z2_2, ....., z2_lt] ]             [ z2]
                            #                   .                                [ . ]
                            #                   .                                [ . ]
                            #         [[zn_1, zn_2, ....., zn_lt] ]              [ zn]

                            #        lt=latent_size
    kl = kl.mean()

    return kl

def loss_function(x, pars, Nf_lognorm, Nf_binomial):
    recon_loss = RecoProb_forVAE(x, pars[0], pars[1], pars[2], Nf_lognorm, Nf_binomial)
    return recon_loss

def RecoProb_forVAE(x, par1, par2, par3, Nf_lognorm, Nf_binomial):

    N = 0
    nll_loss = 0

    #Log-Normal distributed variables
    mu = par1[:,:,:Nf_lognorm]
    sigma = par2[:,:,:Nf_lognorm]
    fraction = par3[:,:,:Nf_lognorm]

    x_clipped = torch.clamp(x[:,:,:Nf_lognorm], clip_x_to0, 1e8)
    single_NLL = torch.where(torch.le(x[:,:,:Nf_lognorm], clip_x_to0),
                            -torch.log(fraction),
                                -torch.log(1-fraction)
                                + torch.log(sigma)
                                + torch.log(x_clipped)
                                + 0.5*torch.mul(torch.div(torch.log(x_clipped) - mu, sigma),
                                                  torch.div(torch.log(x_clipped) - mu, sigma)))
    nll_loss += torch.sum(single_NLL, axis=-1)
    N += Nf_lognorm

    if Nf_binomial != 0:
        #Binomial distributed variables
        p = 0.5*(1+0.98*torch.tanh(par1[:, :,N: N+Nf_binomial]))
        single_NLL = -torch.where(torch.eq(x[:,:, N: N+Nf_binomial],1), torch.log(p), torch.log(1-p))
        nll_loss += torch.sum(single_NLL, axis=-1)
        N += Nf_binomial

    return nll_loss


def RecoProb_forVAE_wrapper(x, pars, Nf_lognorm, Nf_binomial, weights_loss=None):
    recon_loss = RecoProb_forVAE(x, pars[0], pars[1], pars[2], Nf_lognorm, Nf_binomial, weights_loss)
    return recon_loss

def RecoProb_forVAE_weighted(x, par1, par2, par3, Nf_lognorm, Nf_binomial, weights_loss = None):

    N = 0
    nll_loss = 0

    lognorm_weights = [x for x in weights_loss[:Nf_lognorm]]
    binomail_weights = [x for x in weights_loss[:Nf_lognorm]]

    #Log-Normal distributed variables
    for ix, wi in enumerate(lognorm_weights):

        mu = par1[:,ix:ix+1]
        sigma = par2[:,ix:ix+1]
        fraction = par3[:,ix:ix+1]

        x_clipped = torch.clamp(x[:,ix:ix+1], clip_x_to0, 1e8)
        single_NLL = torch.where(torch.le(x[:,ix:ix+1], clip_x_to0),
                                -torch.log(fraction),
                                    -torch.log(1-fraction)
                                    + torch.log(sigma)
                                    + torch.log(x_clipped)
                                    + 0.5*torch.mul(torch.div(torch.log(x_clipped) - mu, sigma),
                                                      torch.div(torch.log(x_clipped) - mu, sigma)))
        nll_loss += torch.sum(wi*single_NLL, axis=-1)

    N += Nf_lognorm

    #Binomial distributed variables

    for ix, wi in enumerate(binomail_weights):
        p = 0.5*(1+0.98*torch.tanh(par1[:, N+ix: N+ix+Nf_binomial]))
        single_NLL = -torch.where(torch.eq(x[:, N: N+Nf_binomial],1), torch.log(p), torch.log(1-p))
        nll_loss += torch.sum(wi*single_NLL, axis=-1)

    N += Nf_binomial

    return nll_loss

def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior, sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    return 0.5 * torch.sum(kl_loss, axis=-1)

#def KL_loss_forVAE(mu, sigma):
#    mu_prior = torch.tensor(0)
#    sigma_prior = torch.tensor(1)
#    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
#    div = torch.div(mu_prior - mu, sigma_prior)
#    kl_loss += torch.mul(div, div)
#    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
#    return 0.5 * torch.sum(kl_loss, axis=-1)

