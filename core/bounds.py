import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import pdb

def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

def h2(y):
    return (1+y)*np.log(1+y) - y

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def hoeffding_minus(mu, x, n):
    return -n * h1(np.minimum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

def bentkus_minus(mu, x, n):
    return np.log(max(binom.cdf(np.ceil(n*x),n,mu),1e-10))+1

def HB_mu_plus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n) 
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

def HB_mu_minus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_minus(mu, muhat, n) 
        bentkus_mu = bentkus_minus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)
