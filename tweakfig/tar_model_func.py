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


def get_sc(h, nk=100):
    """Function to extract the superconducting order"""
    fk = h.get_hk_gen()
    dref = fk(np.random.random(3))[0,2] ; dref = dref/np.abs(dref)
    def f(k):
        m = fk(k)
        return (m[0,2]/dref).real # relevant element of the matrix
    from pyqula import spectrum
    (kxy,sc) = spectrum.reciprocal_map(h,f,nk=nk)
    return sc

# now to it for the two order parameters

def get_band_structure(h, nk=100, en=0, delta=0.1):
    
    nk = 100 # smearing and kmesh
    energies = 0 # energies
    ip = 1 # counter for the plot
    (x,y,d) = h.get_fermi_surface(e=en,delta=Delta,nk=nk) # compute Fermi surface
    return d.reshape((nk,nk)) ; 

def calc_conductance(G, **kwargs):

    _cond = {}
    _cond["total"] = G[:,:,0]/2
    _cond["electron"] = (G[:,:,0] - 2*G[:,:,1])/2
    _cond["Andreev"] = (2*G[:,:,1])/2
    _cond["GA_GT"] = (2*G[:,:,1]/G[:,:,0])
    
    return _cond

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


def plot_sc(h0):

    nk = 100 # number of kpoints
    (kx,ky,fs) = h0.get_fermi_surface(nk=nk,delta=5e-1)
    sc = get_sc(h0, nk)

    z = fs*sc ; z = z/np.max(np.abs(z))
    
    f2,a2 = pplt.subplots()
    a2.scatter(kx,ky,c=z,cmap="bwr",vmin=-0.5,vmax=0.5)
    a2.set_xticks([])
    a2.set_yticks([]) 
    a2.format(xlabel='kx', ylabel="ky")
    #plt.xlabel("kx") ; plt.ylabel("ky") ; plt.colorbar(label="$\\Delta (k)$",ticks=[])
    return f2



def kappa_corr_G(G_p, d2, refwidth=2.5, *args):
    
    es = d2["es"]
    delta = d2["delta"]
    #_roi = slice(5,45,1)
    #G_p = G_T_onsite[5,_roi]

    
    f2,a2 = pplt.subplots(nrows=1,ncols=3, sharey=False, sharex=True, refwidth=refwidth)

    kappa = calc_kappa_all(G_p[:,:,0],G_p[:,0,0])
    kappa_A = calc_kappa_all(G_p[:,:,1],G_p[:,0,0])
    kappa_N = calc_kappa_all((G_p[:,:,0]-2*G_p[:,:,1]),G_p[:,0,0])
    _len = kappa.shape[1]

    plum_cycle = pplt.Cycle("viridis",len(kappa),lw=1.5)
    colors = [c["color"] for c in plum_cycle]

    for _c, kap_T, kap_A, kap_N, _G in zip(colors, kappa, kappa_A,kappa_N, G_p):
        
        ro = 2*_G[:,1]/_G[:,0]
        comb = kap_A*ro+(1-ro)*kap_N

        
        # a2[0].plot(kap_T,kap_N, linewidth=1.5, color=_c)   
        
        # a2[1].plot(es/delta, kap_A,linewidth=1.5, color=_c)
        # a2[2].plot(es/delta, kap_N,linewidth=1.5, color=_c)
        
        #a2[3].plot(es/delta, kap_T,linewidth=1.5, color=_c)

        a2[0].plot(_G[:,0]/_G[0,0],kap_T,'.', linewidth=1.5, color=_c)   
        a2[1].plot(_G[:,1]/_G[0,0],kap_A, '.',linewidth=1.5, color=_c)
        a2[2].plot((_G[:,0]-2*_G[:,1]), kap_N, '.',linewidth=1.5, color=_c)
        
        #a2[7].semilogy(es/delta, ro,label="total",linewidth=1.5, color=_c)
        
        

    x_im = (es/delta)
    y_im = np.arange(len(kappa))

    a2[0].format(title='total',titlesize=14)
    a2[1].format(title='Andreev',titlesize=14)
    a2[2].format(title='e-e',titlesize=14)
    #a2[3].format(title='combined ((1-ro)*e-e + ro*A)',titlesize=14)
    
    # for j in range(len(a2)):
    #     a2[j].format(grid=False,xticklabelsize=12,yticklabelsize=12)
    #     if j > 2:
    #         a2[j].format(yformatter="log")
    

    # a2.format(xlabel="Energy [$\it{\\Delta}$]",xlabelsize=14)
    # a2[0].format(ylabel='$\\kappa/\\kappa_{N}$',ylabelsize=14)
    # a2[3].format(ylabel='exc. cond.',ylabelsize=14)


    return f2
    # a2[2].imshow(kappa, cmap='viridis',extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)
    # a2[5].imshow(np.apply_along_axis(lambda x: (x/x[0]),1,G_p[:,:,1]), 
    #             cmap='viridis', extent=[x_im[0],x_im[-1],y_im[0],y_im[-1]],aspect=1/5)

