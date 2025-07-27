from torch.nn.functional import binary_cross_entropy
from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch

def ELBO_loss(y, t, mu, sigma, beta):
    # Ensure all tensors are on the same device
    device = y.device
    if t.device != device:
        t = t.to(device)
    if mu.device != device:
        mu = mu.to(device)
    if sigma.device != device:
        sigma = sigma.to(device)
    
    # log[p(x|z)]
    likelihood = -binary_cross_entropy(y, t, reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Kullback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # Create normal distributions on the same device
    n_mu = torch.tensor([0.], device=device)
    n_sigma = torch.tensor([1.], device=device)

    p = Normal(n_mu, n_sigma)
    q = Normal(mu, sigma)

    #KL divergence
    kl_div = kl_divergence(q, p)

    # common modification to the ELBO introduces the KL weight hyperparameter β (Bowman et al., 2016; Higgins et al., 2017) 
    elbo = torch.mean(likelihood) - (beta * torch.mean(kl_div))  # Equation (3)


    return -elbo, kl_div.mean(), beta * kl_div.mean()