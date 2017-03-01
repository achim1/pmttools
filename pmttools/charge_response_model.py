"""
Model a PMT charge response in the SPE regime. Everything implemented here
relies heavily on a publication by the BOREXINO colaboration:
see http://ac.els-cdn.com/S0168900200003375/1-s2.0-S0168900200003375-main.pdf?_tid=a4d59bb2-e370-11e6-a363-00000aab0f27&acdnat=1485398542_08fc7a50a2f1945430174767a78edd3

The model has 5 free parameters: 
    - mu_p: The position of the pedestal peak
    - sigma_p: The width of the pedestal peak
    - p: The fraction of events in the exponential part 
         of the charge response spectrum
    - A: The decay constant of the exponential part
    - mu: The position of the SPE peak
    - sigma: The width of the SPE peak

"""

import pyosci.tools as tools
import pyosci.plotting as plt

from pyevsel import fitting as fit

import numpy as np

from scipy.special import erf

from functools import reduce

from . import characteristics as c

# The individual contributions to the model

def pedestal(x, mu_p, sigma_p, p, A, mu, sigma): 
    """
    The pedestal part of a SPE charge spectrum

    """

    return fit.gauss(x, mu_p, sigma_p, 1)

#####################################################################

def single_PE_response(x, mu_p, sigma_p, p, A, mu, sigma):
    """
    The SPE charge response
    """   
    # I am not sure here, according to the paper it would be sqrt(sigma)
    # however, how this is defined it makes no sense here
    # What about erf then? Is it wrong?
    spe_0 = fit.gauss((x - (mu_p*np.ones(len(x)))),mu, sigma,1)
    #spe_0 = (1/np.sqrt(2*sigma*np.pi))
    spe_0 *= (1 - p)/(0.5*(1 + erf(mu/(np.sqrt(2)*sigma))))
    
    # only use in the case of no folding
    #result[x <= mu_p] = 0
    return spe_0

##########################################################

def two_PE_response(x, mu_p, sigma_p, p, A, mu, sigma):
    """
    The two photo electron peak described more precisely

    Args:
        x (np.ndarray): charges 
        *args: fitparams
    """
    term1 = (p**2)*((x-mu_p)/(A**2))*np.exp(-((x-mu_p)/A))
    term2 = 2*((1-p)*p)/(np.sqrt(2*np.pi)*sigma)
    term2_exp = np.exp(-0.5*(((x - mu_p - mu - A)/sigma)**2)) 
    term3 = ((1-p)**2)/(2*np.sqrt(np.pi)*sigma)
    term3_exp = np.exp(-0.5*(((x - mu_p - (2*mu))/(sigma*np.sqrt(2)))**2))
    return (term1 + (term2*term2_exp) + (term3*term3_exp))

###########################################################

def convolve_exponential_part(lower_bound, upper_bound):
    """
    Factory function to Construct the convolution
    of the exponential part and the gaussian part
    within the limits of the fit
    
    Args:
        L (float): lower integration bound
        U (float): upper integration bound

    """
    def convolved_exp(x,  mu_p, sigma_p, p, A, mu, sigma):
        """
        Result from Maple
        """
        
        prefactor = (p*(2**(3./4.)))/(4*A)
        exponent = (np.sqrt(2)*(sigma_p**2)) + (8*A*mu_p) - (4*A*x)
        exponent /= 4*(A**2)
        erf1 = ((2**(1/4.)))*((np.sqrt(2)*lower_bound*A)-(np.sqrt(2)*mu_p*A) - (np.sqrt(2)*A*x) + (sigma_p**2))
        erf1/= 2*sigma_p*A
        erf2 = ((2**(1/4.)))*((np.sqrt(2)*upper_bound*A)+(np.sqrt(2)*mu_p*A) + (sigma_p**2))
        erf2 /= 2*sigma_p*A
        #print (-erf(erf1) + erf(erf2))
        return prefactor*np.exp(exponent)*(-erf(erf1) + erf(erf2))

    return convolved_exp

##################################################

def simple_exponential_response(x,  mu_p, sigma_p, p, A, mu, sigma):
    """
    Exponential part of the SPE charge response
    """
    exponential = (p/A)*np.exp(-1*(x - mu_p)/A)
    exponential[x>=(mu+sigma)] = 0
    exponential[x < 0] = 0
    return exponential

##############################################################

def multi_PE_response(n):
    """
    Calculate the multi photoelectron response for the n-th PE.
    Note: in general, linearity is assumed for n>2. 

    Args:
        n (int): number of PE to calculate the response for

    Returns:
        callable

    """
    def m_i(x,  mu_p, sigma_p, p, A, mu, sigma):
        # no fit here
        mu_m = ((1 - p)*mu) + (p*A)
        sigma_m = ((1-p)*(sigma**2 + mu**2)) + (2*p*(A**2)) - (mu_m**2)
        response = fit.gauss(x - mu_p, mu_m, sigma_m, n)
        return response
    return m_i

###########################################################

def construct_charge_response_model(charges,\
                                    lowest_mpe_contrib=3,\
                                    highest_mpe_contrib=6,\
                                    model_2PE_response=True,
                                    convolved_exponential=True,
                                    nbins=200):
    """
    Calculate from the data the possibilities to observe
    None, 1 and more pe with poisson prob.
    Put together the model and attach these values
    
    Args:
        charges (np.ndarray): Integrated charges of measured wavefomrs
    
    Keyword Args:
        lowest_mpe_contrib (int): lowest multi PE contribution (has to be at lest 3)
        highest_mpe_contrib (int): up to which contribution should be fitted
        model_2PE_response (bool): Use a modified gauss to model the 2PE peak 
        convolved_exponential (bool): Convolve the exponential part with the pedestal
        nbins (int): Number of bins to use for the histogram
    """
        
    n_hit, n_all = c.get_n_hit(charges, nbins)
    mu_exp = c.calculate_mu(charges, nbins)

    # the model needs to know about how likely
    # the individual occurences are
    # simpy attach them to the functions here
    p0 = np.exp(-mu_exp)
    p1 = np.exp(-mu_exp)*mu_exp
    p2 = fit.poisson(mu_exp, 2)

    model = fit.Model(pedestal, func_norm=p0)
    if convolved_exponential:
        exponential_term = convolve_exponential_part(min(charges),max(charges))
    else:
        exponential_term = simple_exponential_response
    model += fit.Model(exponential_term, func_norm=p1)
    model += fit.Model(single_PE_response, func_norm=p1)

    # treat the second (2PE) peak differently
    if model_2PE_response:
        model += fit.Model(two_PE_response, func_norm=p2)
        if lowest_mpe_contrib == 2:
            lowest_mpe_contrib = 3
    # add a multi pe response
    for k, mpe_mod in enumerate([multi_PE_response(n) for n in range(lowest_mpe_contrib, highest_mpe_contrib)]):
        mpe_norm = fit.poisson(mu_exp,k+lowest_mpe_contrib)
        model += fit.Model(mpe_mod, func_norm=mpe_norm)

    model.couple_all_models()
    print("Calculated mu of", mu_exp)
    print("Got n_hit ", n_hit, "n_nohit ",n_all - n_hit)
    return model



