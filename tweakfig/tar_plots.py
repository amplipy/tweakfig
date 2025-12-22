import numpy as np
#import proplot as pplt
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter as savgol

from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, FixedFormatter


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def colorFader(c1, c2, mix=0):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def calc_conductance(G, **kwargs):

    _cond = {}
    _cond["total"] = G[:,:,0]/2
    _cond["electron"] = (G[:,:,0] - 2*G[:,:,1])/2
    _cond["Andreev"] = (2*G[:,:,1])/2
    _cond["GA_GT"] = (2*G[:,:,1]/G[:,:,0])
    
    return _cond

def calc_kappa(G_T, en_ind=30, _plot=0):
    from scipy.signal import savgol_filter as savgol
    #assumes conductance array in dimensions of coupling (0) vs energy (1)
    x = G_T[:,0]
    y = G_T[:,en_ind]
    
    dlogx = savgol(np.log(x),window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(y),window_length=5, polyorder=2, deriv=1)
    
    kappa = dlogy/dlogx

    if _plot:
        f2,a2 = pplt.subplots()
        a2.plot(x,kappa)

def calc_kappa_all(G_T, x, thresh=1e-4):
    from scipy.signal import savgol_filter as savgol
    #assumes conductance array in dimensions of coupling (0) vs energy (1)
    #so it's dlog(G(z,E))/dlog(G[z,0])
    
    dlogx = savgol(np.log(x),axis=0, window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(G_T),axis=0, window_length=5, polyorder=2, deriv=1)
    kappa = np.apply_along_axis(lambda x: x/dlogx, 0, dlogy)

    return np.where(G_T>thresh, kappa, np.nan)

def calc_true_kappa(G_T, x):
    from scipy.signal import savgol_filter as savgol
    #this one assumes that x is the TRUE coupling coefficient.
    #this can be t^2 from calculations
    #or maybe experimental values of z
    #so it's dlog(G)/dZ
    
    dlogx = savgol(x,axis=0, window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(G_T),axis=0, window_length=5, polyorder=2, deriv=1)
    unnormalized = np.apply_along_axis(lambda x: x/dlogx, 0, dlogy)
    kappa_kappa_n = np.apply_along_axis(lambda y: y/y[0], 1, unnormalized)
    return kappa_kappa_n


_nred = lambda x: x/x[0]

def plot_conductance(G_plot, d2, **kwargs):

    #G_plot = G_T[0:30]
    es = d2["es"]
    delta = d2["delta"]

    if "figax" not in kwargs.keys():
        f2, a2 = plt.subplots(1, 3, sharey=False, figsize=(10,3))
    else:
        f2, a2 = kwargs["figax"]
    


    # Apply 'viridis' colormap for coloring the spectra
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0.1, 1,len(G_plot)))

    for _c, G in zip(colors, G_plot):

        a2[0].semilogy(es/delta, G[:,0],label="total",linewidth=1.5, color=_c)
        
        # # now each component
        a2[1].semilogy(es/delta,G[:,0]-2*G[:,1],label="electron", color=_c)
        #a2[1].plot(es/delta,2*G[:,1],'--',label="Andreev", color=_c)
        a2[1].plot(es/delta,2*G[:,1],label="Andreev", color='red', alpha=0.2)
        
        a2[2].semilogy(es/delta,2*G[:,1]/G[:,0],'-',label="Andreev", color=_c, alpha=0.4)
        

    #for j in np.arange(len(args)):
    #    a2[j].set_ylim(args[j])
        
    #a2[0].format(grid=False,xticklabelsize=12,yticklabelsize=12,yformatter="log")
    if "noxlabel" not in kwargs.keys():
        a2[0].set_xlabel("Energy ($\it{\\Delta}$)", labelpad=3.0)
        a2[1].set_xlabel("Energy ($\\Delta$)", labelpad=3.0)
        a2[2].set_xlabel("Energy ($\\Delta$)", labelpad=3.0)

    a2[0].set_ylabel("$G \\ (\it{G_{0}})$", labelpad=3.0)
    a2[1].set_ylabel("$G \\ (\it{G_{0}})$", labelpad=3.0)
    a2[2].set_ylabel("$G_{A}/G$", labelpad=3.0)

    plt.tight_layout()
    if "figax" not in kwargs.keys():
        return f2

def plot_kappas(G_p, d2, refwidth=2.5, **kwargs):
    
    #_roi = slice(5,45,1)
    #G_p = G_T_onsite[5,_roi]

    
    if "figax" not in kwargs.keys():
        f2,a2 = plt.subplots(1,3, figsize=(10,3))
    else:
        f2, a2 = kwargs["figax"]

    res = {}

    es = d2["es"]
    delta = d2["delta"]
    
    res["kappa"] = kappa = calc_kappa_all(G_p[:,:,0],G_p[:,0,0])
    res["kappa_A"] = kappa_A = calc_kappa_all(G_p[:,:,1],G_p[:,0,0])
    res["kappa_N"] = kappa_N = calc_kappa_all((G_p[:,:,0]-2*G_p[:,:,1]),G_p[:,0,0])
    _len = kappa.shape[1]

    # Apply 'viridis' colormap for coloring the spectra
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0.1, 1,len(G_p)))

    for _c, kap_T, kap_A, kap_N, _G in zip(colors, kappa, kappa_A,kappa_N, G_p):
        
        ro = 2*_G[:,1]/_G[:,0]
        comb = kap_A*ro+(1-ro)*kap_N

        
        a2[0].plot(es/delta, kap_T,linewidth=1.5, color=_c)   
        a2[0].plot((es/delta)[int(_len/2):], comb[int(_len/2):],'--', color='red', linewidth=1.5, alpha=0.5)
        
        a2[1].plot(es/delta, kap_A,linewidth=1.5, color=_c)
        a2[2].plot(es/delta, kap_N,linewidth=1.5, color=_c)
        

    a2[0].set_xlabel("Energy ($\it{\\Delta}$)", labelpad=3.0)
    a2[0].set_ylabel("$\kappa$", labelpad=3.0)

    #a2[1].format(grid=False,xticklabelsize=12,yticklabelsize=12,yformatter="log")
    a2[1].set_xlabel("Energy ($\\Delta$)", labelpad=3.0)
    a2[1].set_ylabel("$\kappa_{A}$", labelpad=3.0)

    #a2[2].format(grid=False,xticklabelsize=12,yticklabelsize=12,yformatter="log")
    a2[2].set_xlabel("Energy ($\\Delta$)", labelpad=3.0)
    a2[2].set_ylabel("$\kappa_{e}$", labelpad=3.0)
    
    plt.tight_layout()
    #x_im = (es/delta)
    #y_im = np.arange(len(kappa))


    #a2[0].format(title='total',titlesize=14)
    #a2[1].format(title='Andreev',titlesize=14)
    #a2[2].format(title='e-e',titlesize=14)
    #a2[3].format(title='combined ((1-ro)*e-e + ro*A)',titlesize=14)
    
    # for j in range(len(a2)):
    #     a2[j].format(grid=False,xticklabelsize=12,yticklabelsize=12)
    #     if j > 2:
    #         a2[j].format(yformatter="log")
    

    # a2.format(xlabel="Energy [$\it{\\Delta}$]",xlabelsize=14)
    # a2[0].format(ylabel='$\\kappa/\\kappa_{N}$',ylabelsize=14)
    # a2[3].format(ylabel='exc. cond.',ylabelsize=14)


    return [f2, res]
    # a2[2].imshow(kappa, cmap='viridis',extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)
    # a2[5].imshow(np.apply_along_axis(lambda x: (x/x[0]),1,G_p[:,:,1]), 
    #             cmap='viridis', extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)


def plot_kappas_old(G_p, d2, refwidth=2.5, *args):
    
    #_roi = slice(5,45,1)
    #G_p = G_T_onsite[5,_roi]

    
    res = {}

    es = d2["es"]
    delta = d2["delta"]

    f2,a2 = pplt.subplots(nrows=2,ncols=3, sharey=False, sharex=True, refwidth=refwidth)

    res["kappa"] = kappa = calc_kappa_all(G_p[:,:,0],G_p[:,0,0])
    res["kappa_A"] = kappa_A = calc_kappa_all(G_p[:,:,1],G_p[:,0,0])
    res["kappa_N"] = kappa_N = calc_kappa_all((G_p[:,:,0]-2*G_p[:,:,1]),G_p[:,0,0])
    _len = kappa.shape[1]

    plum_cycle = pplt.Cycle("viridis",len(kappa),lw=1.5)
    colors = [c["color"] for c in plum_cycle]

    for _c, kap_T, kap_A, kap_N, _G in zip(colors, kappa, kappa_A,kappa_N, G_p):
        
        ro = 2*_G[:,1]/_G[:,0]
        comb = kap_A*ro+(1-ro)*kap_N

        
        a2[0].plot(es/delta, kap_T,linewidth=1.5, color=_c)   
        a2[0].plot((es/delta)[int(_len/2):], comb[int(_len/2):],'--', color='red', linewidth=1.5, alpha=0.5)
        
        a2[1].plot(es/delta, kap_A,linewidth=1.5, color=_c)
        a2[2].plot(es/delta, kap_N,linewidth=1.5, color=_c)
        
        #a2[3].plot(es/delta, kap_T,linewidth=1.5, color=_c)

        a2[3].semilogy(es/delta, _G[:,0]/_G[0,0],linewidth=1.5, color=_c)   
        a2[4].semilogy(es/delta, _G[:,1]/_G[0,1],linewidth=1.5, color=_c)
        a2[5].semilogy(es/delta, (_G[:,0]-2*_G[:,1])/_G[0,0],linewidth=1.5, color=_c)
        
        #a2[7].semilogy(es/delta, ro,label="total",linewidth=1.5, color=_c)
        
        

    x_im = (es/delta)
    y_im = np.arange(len(kappa))

    a2[0].format(title='total',titlesize=14)
    a2[1].format(title='Andreev',titlesize=14)
    a2[2].format(title='e-e',titlesize=14)
    #a2[3].format(title='combined ((1-ro)*e-e + ro*A)',titlesize=14)
    
    for j in range(len(a2)):
        a2[j].format(grid=False,xticklabelsize=12,yticklabelsize=12)
        if j > 2:
            a2[j].format(yformatter="log")
    

    a2.format(xlabel="Energy [$\it{\\Delta}$]",xlabelsize=14)
    a2[0].format(ylabel='$\\kappa/\\kappa_{N}$',ylabelsize=14)
    a2[3].format(ylabel='exc. cond.',ylabelsize=14)


    return [f2, res]
    # a2[2].imshow(kappa, cmap='viridis',extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)
    # a2[5].imshow(np.apply_along_axis(lambda x: (x/x[0]),1,G_p[:,:,1]), 
    #             cmap='viridis', extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)
