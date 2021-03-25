import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
import pdb

def safe_min(x):
    if np.any(np.isnan(x)):
        return -np.Inf
    else:
        return x.min()

# replicates the functionality of R's expand.grid
def expand_grid(arr1,arr2):
    params = np.meshgrid(arr2,arr1)
    return params[1].T.flatten(), params[0].T.flatten()

def normalized_vapnik_tail_up(n, m, alpha, eta, maxiter):
    c1 = np.log(1 / 4 / (1-stats.norm.cdf(np.sqrt(2))) )
    c2 = 5 * np.sqrt( 2*np.pi*np.exp(1) ) * ( 2*stats.norm.cdf(1) - 1)
    def _tailprob(x):
        gamma, n_p = expand_grid(np.arange(0.001, 0.5 + 0.001, 0.001), np.arange(0.5, 3 + 0.1, 0.1))
        n_p = np.ceil(n**(n_p))
        kappa = eta + x**2/2 + x*np.sqrt(x**2/4 + eta)
        kappa = eta + (gamma + n/n_p) / (1 + n/n_p) * np.sqrt(kappa) 
        fac1 = 1 - np.exp(-n_p*x**2/2 * gamma**2/(1 + gamma**2*x**2/36/eta))
        fac2 = 1 - (np.sqrt(1+eta) - np.sqrt(eta))**2 / (n_p * x**2 * gamma**2)
        log_denom = np.log(np.maximum(0,fac1,fac2))

        g2 = n/(1 + n/n_p)**2  * x**2/2 * (1 - gamma)**2/(1 + (1 - gamma)**2 * x**2 / 36 / kappa)
        log_Delta = np.log(m*(n + n_p) + 1)
        log_prob_bardenet = safe_min(log_Delta - g2 - log_denom)

        tmp = np.sqrt(n * (1 + eta)/2) * (1-gamma) * x
        gauss = 1-stats.norm.cdf(tmp)
        extra_term = np.log(2*n*m + 1) - log_denom
        log_prob_bentkus_dzindzalieta = safe_min(c1 + np.log(gauss) + extra_term)
        log_prob_pinelis = safe_min(np.log(gauss + stats.norm.pdf(tmp)/(9+tmp**2)*c2) + extra_term)
        log_prob_hoeffding = safe_min(-tmp**2/2 + extra_term)
        log_prob = np.min([log_prob_bardenet, log_prob_bentkus_dzindzalieta, log_prob_pinelis, log_prob_hoeffding])
        return log_prob - np.log(alpha)

    return brentq(_tailprob,0,1,maxiter=maxiter) 

if __name__ == "__main__":
    n = 10000
    m = 9
    alpha = 0.2
    eta = 0.1
    maxiter = 100000
    print(normalized_vapnik_tail_up(n, m, alpha, eta, maxiter))
