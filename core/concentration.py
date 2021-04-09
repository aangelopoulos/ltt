import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq
import pdb
from pathlib import Path

CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

def safe_min(x):
    if np.any(np.isnan(x)):
        return -np.Inf
    else:
        return x.min()

# replicates the functionality of R's expand.grid
def expand_grid(arr1,arr2):
    params = np.meshgrid(arr2,arr1)
    return params[1].T.flatten(), params[0].T.flatten()

# Get t
# Equation 12 in Lihua's note 
def normalized_vapnik_tail_upper(n, m, delta, eta, maxiter):
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
        return log_prob - np.log(delta)

    return brentq(_tailprob,0,1,maxiter=maxiter) 

# Equation 11 in Lihua's note.
def normalized_vapnik_tail_lower(n, m, delta, eta, maxiter):
    c1 = np.log(1 / 4 / (1-stats.norm.cdf(np.sqrt(2))) )
    c2 = 5 * np.sqrt( 2*np.pi*np.exp(1) ) * ( 2*stats.norm.cdf(1) - 1)
    def _tailprob(x):
        gamma, n_p = expand_grid(np.arange(0.001, 0.5 + 0.001, 0.001), np.arange(0.5, 3 + 0.1, 0.1))
        n_p = np.ceil(n**(n_p))
        kappa_plus = eta + x**2/2 + x*np.sqrt(x**2/4 + eta)
        kappa_minus = eta + (gamma + n/n_p) / (1 + n/n_p) * np.sqrt(kappa_plus) 
        fac1 = 1 - np.exp(-n_p*x**2/2 * gamma**2/(1 + gamma**2*x**2/36/kappa_plus))
        fac2 = 1 - (np.sqrt(1+kappa_plus) - np.sqrt(kappa_plus))**2 / (n_p * x**2 * gamma**2)
        log_denom = np.log(np.maximum(0,fac1,fac2))

        g2 = n/(1 + n/n_p)**2  * x**2/2 * (1 - gamma)**2/(1 + (1 - gamma)**2 * x**2 / 36 / eta)
        log_Delta = np.log(m*(n + n_p) + 1)
        log_prob_bardenet = safe_min(log_Delta - g2 - log_denom)

        tmp = np.sqrt(n * (1 + eta)/2) * (1-gamma) * x
        gauss = 1-stats.norm.cdf(tmp)
        extra_term = np.log(2*n*m + 1) - log_denom
        log_prob_bentkus_dzindzalieta = safe_min(c1 + np.log(gauss) + extra_term)
        log_prob_pinelis = safe_min(np.log(gauss + stats.norm.pdf(tmp)/(9+tmp**2)*c2) + extra_term)
        log_prob_hoeffding = safe_min(-tmp**2/2 + extra_term)
        log_prob = np.min([log_prob_bardenet, log_prob_bentkus_dzindzalieta, log_prob_pinelis, log_prob_hoeffding])
        return log_prob - np.log(delta)

    return brentq(_tailprob,0,1,maxiter=maxiter) 

# Return upper bound fdr
def shat_upper_tail(s, n, m, delta, eta, maxiter):
    t = normalized_vapnik_tail_upper(n, m, delta, eta, maxiter)
    def _condition(shat):
        return (shat - s)/np.sqrt(shat + eta) - t
    shat = brentq(_condition,0,1,maxiter=maxiter) 
    return shat

def shat_lower_tail(s, n, m, delta, eta, maxiter):
    t = normalized_vapnik_tail_lower(n, m, delta, eta, maxiter)
    def _condition(shat):
        return (s - shat)/np.sqrt(s + eta) - t
    try:
        shat = brentq(_condition,0,1,maxiter=maxiter) 
    except:
        print("Warning: setting \hat{s} to 0 due to failed search")
        shat = 0
    return shat

# General upper and lower bounds
def nu_plus(n, m, nu, delta, maxiter):
    eta_star = get_eta_star_upper(n, m, nu, delta, maxiter)
    t = normalized_vapnik_tail_upper(n, m, delta, eta_star, maxiter)
    def _condition(nu_plus):
        return nu - (nu_plus + t * np.sqrt(nu_plus + eta_star))
    try:
        nu_plus = brentq(_condition,0,1,maxiter=maxiter)
    except:
        print("Warning: setting alpha_plus to 0 due to failed search")
        nu_plus = 0
    return nu_plus 

def r_minus(n, m, r, delta, maxiter):
    eta_star = get_eta_star_upper(n, m, r, delta, maxiter)
    t2 = normalized_vapnik_tail_lower(n, m, delta, eta_star, maxiter)
    r_minus = r - t2*np.sqrt(r+eta_star)
    return r_minus 

# Return required fdp needed to achieve an fdr of alpha
def required_fdp(n, m, alpha, delta, maxiter):
    return nu_plus(n, m, alpha, delta, maxiter)
#    eta_star = get_eta_star_upper(n, m, alpha, delta, maxiter)
#    t = normalized_vapnik_tail_upper(n, m, delta, eta_star, maxiter)
#    def _condition(alpha_plus):
#        return alpha - (alpha_plus + t * np.sqrt(alpha_plus + eta_star))
#    try:
#        alpha_plus = brentq(_condition,0,1,maxiter=maxiter)
#    except:
#        print("Warning: setting alpha_plus to 0 due to failed search")
#        alpha_plus = 0
#    return alpha_plus 

def pfdr_ucb(n, m, accuracy, frac_abstention, delta, maxiter):
    nu_plus = nu_plus(n, m, 1-accuracy, delta, maxiter)
    r_minus = r_minus(n, m, frac_abstention, delta, maxiter)
    if nu_plus <= 0 and r_minus <= 0:
        return 0
    if nu_plus > 0 and r_minus <= 0:
        return np.Inf
    return nu_plus/r_minus

# Get optimal eta for upper tail
def get_eta_star_upper(n, m, alpha, delta, maxiter):
    fname = f'eta_star_{n}_{alpha}_{delta}'
    fpath = CACHE+fname+'.npy'
    if os.path.exists( fpath ):
        return np.load( fpath )
    else:
        eta_grid = np.logspace(-5,1,100)
        best_x = 0
        eta_star = 1
        for eta in eta_grid:
            t = normalized_vapnik_tail_upper(n, m, delta, eta, maxiter)
            x = 0.5*(t*np.sqrt(4*alpha+4*eta+t*t) + 2*alpha + t*t)
            if x >= best_x:
                best_x = x
                eta_star = eta
        os.makedirs(CACHE, exist_ok=True)
        np.save( fpath, eta_star )
        return eta_star

if __name__ == "__main__":
    s = 0.2
    ns = np.logspace(2,5,100)
    m = 1000
    delta = 0.2
    alpha = 0.1
    maxiter = 100000
    eta_star_upper = get_eta_star_upper(ns[0], m, alpha, delta, maxiter)

    # Plot upper tail
    shats = [shat_upper_tail(s, n, m, delta, eta_star_upper, maxiter) for n in ns]
    plt.plot(ns,shats)
    plt.xscale('log')
    plt.savefig('../outputs/shat_upper_tail.pdf')