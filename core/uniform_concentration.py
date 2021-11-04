import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq
import pdb
from pathlib import Path
import pickle as pkl

CACHE = str(Path(__file__).parent.absolute()) + '/.cache/'

def cacheable(func):
    def cache_func(*args):
        fname = CACHE + str(func).split(' ')[1] + str(args) + '.pkl'
        os.makedirs(CACHE, exist_ok=True)
        try:
            filehandler = open(fname, 'rb')
            result = pkl.load(filehandler)
            return result 
        except:
            filehandler = open(fname, 'wb')
            result = func(*args)
            pkl.dump(result, filehandler)   
            return result

    return cache_func

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
def normalized_vapnik_tail_upper(n, m, delta, eta, maxiter,num_grid_points=None):
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
        # If the grid is fixed, log_Delta changes.
        if num_grid_points != None:
            log_Delta = np.log(num_grid_points)
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
def normalized_vapnik_tail_lower(n, m, delta, eta, maxiter, num_grid_points=None):
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
        # If the grid is fixed, log_Delta changes.
        if num_grid_points != None:
            log_Delta = np.log(num_grid_points)
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
def shat_upper_tail(s, n, m, delta, eta, maxiter, num_grid_points=None):
    t = normalized_vapnik_tail_upper(n, m, delta, eta, maxiter, num_grid_points=num_grid_points)
    def _condition(shat):
        return (shat - s)/np.sqrt(shat + eta) - t
    shat = brentq(_condition,0,1,maxiter=maxiter) 
    return shat

def shat_lower_tail(s, n, m, delta, eta, maxiter, num_grid_points=None):
    t = normalized_vapnik_tail_lower(n, m, delta, eta, maxiter, num_grid_points=num_grid_points)
    def _condition(shat):
        return (s - shat)/np.sqrt(s + eta) - t
    try:
        shat = brentq(_condition,0,1,maxiter=maxiter) 
    except:
        print("Warning: setting \hat{s} to 0 due to failed search")
        shat = 0
    return shat

# General upper and lower bounds
@cacheable
def nu_plus(n, m, nu, alpha, delta, maxiter, num_grid_points):
    eta_star = get_eta_star_upper(n, m, alpha, delta, 20, num_grid_points=num_grid_points)
    t = normalized_vapnik_tail_upper(n, m, delta, eta_star, maxiter, num_grid_points=num_grid_points)
    try:
        nu_plus = nu + t*np.sqrt(nu + eta_star + (t*t)/4) + (t*t)/2 
        nu_plus = min(max(nu_plus,0),1)
    except:
        print("Warning: setting nu_plus to 1 due to failed search")
        nu_plus = 1 
    return nu_plus 

# The empirical risk required to get an upper bound of b
def required_empirical_risk(b, n, m, alpha, delta, maxiter, num_grid_points):
    def _condition(er):
        return (nu_plus(n, m, er, alpha, delta, maxiter, num_grid_points) - b)
    try:
        return brentq(_condition, 0, 1, maxiter=maxiter)
    except:
        return 0

@cacheable
def r_minus(n, m, r, alpha, delta, maxiter, num_grid_points):
    eta_star = get_eta_star_upper(n, m, alpha, delta, 20, num_grid_points=num_grid_points)
    t2 = normalized_vapnik_tail_lower(n, m, delta, eta_star, maxiter, num_grid_points=num_grid_points)
    r_minus = r - t2*np.sqrt(max(r+eta_star, 0))
    return r_minus 

# Get optimal eta for upper tail
def get_eta_star_upper(n, m, alpha, delta, maxiter, num_grid_points=None):
    alpha = np.round(alpha,2)
    delta = np.round(alpha,2)
    fname = f'eta_star_{n}_{alpha:.2f}_{delta:.2f}'
    fpath = CACHE+fname+'.npy'
    if os.path.exists( fpath ):
        eta_star = np.load( fpath )
    else:
        print(f"Computing eta_star for {n}, {alpha:.2f}, {delta:.2f}")
        eta_grid = np.logspace(-20,1,50)
        best_x = 0
        eta_star = 1
        for eta in eta_grid:
            try:
                t = normalized_vapnik_tail_upper(n, m, delta, eta, 20, num_grid_points=num_grid_points)
                x = alpha - t*np.sqrt(alpha + eta) 
                if x >= best_x:
                    best_x = x
                    eta_star = eta
            except:
                pass
        print(f"ETA STAR: {eta_star}")
        os.makedirs(CACHE, exist_ok=True)
        np.save( fpath, eta_star )

    print(f"Eta star: {eta_star}")
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
    plt.savefig('../outputs/concentration_results/shat_upper_tail.pdf')
