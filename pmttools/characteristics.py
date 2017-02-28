"""
Some characteristica of the pmt charge response.

"""

from scipy.constants import elementary_charge as ELEMENTARY_CHARGE

def chi2_exponential_part(xs, data, pred, mu_ped, mu_ser):
    """
    The charge response spectrum typically has a fraction of events with 
    charges between 0 and 1 PE which are mostly best described by 
    some kind of exponential fit.
    This function calculates the chi2 for this part.

    """
    mask = np.logical_and(xs >= mu_ped, xs <= mu_ser)
    data = data[mask]
    pred = pred[mask]
    xs   = xs[mask]
    chi2_ndf = calculate_chi_square(data, pred)/len(xs)
    return chi2_ndf

#####################################################

def calculate_peak_to_valley_ratio(bestfitmodel, mu_ped, mu_spe, control_plot=False):
    """
    Calculate the peak to valley ratio
    Args:
        bestfitmodel (fit.Model): A fitted model to charge response data
        mu_ped (float): The x value of the fitted pedestal
        mu_spe (flota): The x value of the fitted spe peak
   
    Keyword Args:
        control_plot (bool): Show control plot to see if correct values are found
   
    """
    

    tmpdata = bestfitmodel.prediction(bestfitmodel.xs)
    valley = min(tmpdata[np.logical_and(bestfitmodel.xs > mu_ped,\
                                    bestfitmodel.xs < mu_spe)])
    valley_x = bestfitmodel.xs[tmpdata == valley]

    peak = max(tmpdata[bestfitmodel.xs > valley_x])
    peak_x = bestfitmodel.xs[tmpdata == peak]
    peak_v_ratio = (peak/valley)
    
    if control_plot:
        fig = p.figure()
        ax = fig.gca()
        ax.plot(bestfitmodel.xs,tmpdata)
        print (valley)
        print (valley_x)

        ax.scatter(valley_x,valley,marker="o")
        ax.scatter(peak_x, peak, marker="o")
        ax.set_ylim(ymin=1e-4)
        ax.set_yscale("log")
        ax.grid(1)
        sb.despine()

    return peak_v_ratio

################################################

def get_n_hit(charges, nbins):
    """
    Identify how many events are in the pedestal of a charge response 
    spectrum.

    Args:
        charges (np.ndarray): The measured charges
        nbins (int): number of bins to use

    Returns:
        tuple (n_hit, n_all)
    """
    one_gauss = lambda x, n, y, z: n * gauss(x, y, z, 1)
    ped_mod = Model(one_gauss, (1000, -.1, 1))
    ped_mod.add_data(charges, nbins=nbins, normalize=False, create_distribution=True )
    ped_mod.fit_to_data(silent=True)
    n_hit = abs(ped_mod.data - ped_mod.prediction(ped_mod.xs)).sum()
    n_pedestal = ped_mod._distribution.stats.nentries - n_hit
    n_all = ped_mod._distribution.stats.nentries
    return n_hit, n_all

##############################################

def calculate_gain(mu_ped, mu_spe, prefactor=1e-12):
    """
    Calculate the pmt gain from the charge distribution

    Args:
        mu_ped (float): the mean of the gaussian fitting the pedestal
        mu_spe (float): the mean of the gaussian fitting the spe response

    Keyword Args:
        prefactor (float): unit of charge (default pico coulomb)

    Returns:
        float
    """
    charge = abs(mu_spe) - abs(mu_ped)
    charge *= prefactor
    return charge/ELEMENTARY_CHARGE



