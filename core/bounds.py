import numpy as np
from scipy.stats import binom, norm
from scipy.optimize import brentq
from confseq import betting
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

def binom_p_value(r_hat,n,alpha):
    return binom.cdf(np.ceil(n*r_hat),n,alpha)

def hb_p_value(r_hat,n,alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat),n,alpha)
    def h1(y,mu):
        with np.errstate(divide='ignore'):
            return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(min(r_hat,alpha),alpha))
    return min(bentkus_p_value,hoeffding_p_value)

def wsr_p_value(losses, alpha, delta=0.1):
    n = len(losses)
    _wsr_lbda_fns_positive = lambda x, m: betting.lambda_predmix_eb(x, alpha=delta, fixed_n=n)
    _wsr_lbda_fns_negative = None

    martingale = betting.betting_mart(losses,alpha,delta,lambdas_fn_positive=_wsr_lbda_fns_positive,lambdas_fn_negative=_wsr_lbda_fns_negative)
    pval = (1/martingale).min()

    return pval

def clt_p_value(r_hat, sigma_hat, n, alpha):
    pval = norm.cdf((np.sqrt(n)*(r_hat-alpha))/sigma_hat)
    return pval

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
    pdb.set_trace()
    if _tailprob(1e-10) > 0:
        return 0
    else:
        return brentq(_tailprob, 1e-10, muhat, maxiter=maxiters)

if __name__ == "__main__":
    print(HB_mu_minus(0.5, 100, 0.1, 1000))
