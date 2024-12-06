import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scipy.stats as st
from math import exp,sqrt,log
import matplotlib.pyplot as plt
from arviz import kde
import scienceplots

def load_sample_data(filename,id_col):
    """    
    File format for sampled data
    id_col = (1, 2 or 3)
    """
    data = np.loadtxt(filename, comments="#")
    tgt  = data[:,id_col]
    x = np.linspace(tgt.min(),tgt.max(),len(tgt))
    return x , tgt

def coeff_var_ln(std):
    aux = 100.0*exp(2.0 * log(10.0) * std**2.0) - 1.0
    return sqrt(aux)

def coeff_var(mean,std):
    return 100.0 * (std/mean)

def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    """ https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python/37616966#37616966"""
    x , y = data
    DISTRIBUTIONS = [st.norm]
    #DISTRIBUTIONS = [st.norm,st.beta,st.gamma,st.lognorm,st.rayleigh,st.t]#,st.uniform,st.expon]
#    DISTRIBUTIONS = [        
#        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
#        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
#        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
#        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
#        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
#        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
#        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
#        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
#        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
#        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
#    ]

    # Best holders
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # fit dist to data
        params = distribution.fit(y)
        #print("Testing distribution:",distribution.name)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.power((y - pdf) / (np.max(y)), 2.0) 
        sse = np.sum(sse)
        
        #print('Distribution::',distribution.name,'Root Mean Square error::',np.sqrt(sse) / (np.power(len(y),2)))
        #print('Dist. params::',params)
        #print('')

    # identify if this distribution is better
        if (best_sse > sse):
            best_distribution = distribution
            best_params = params
            best_sse = sse
            best_pdf = pdf

    return best_distribution.name, best_params, best_pdf

########################################################################
# The execution
########################################################################

if __name__ == "__main__":
    plt.style.use('science')

    labels_name = ['fmmob','SF','sfbet','epcap']
    labels      = ['$fmmob$','$SF$','$sfbet$','$epcap$']
    file_id     = 'stars_mcmc.arv'

    limits = [[200, 1100],
              [0.45, 0.47],
              [10, 400],
              [0.2, 0.8]]
    
    # Now PDFs for the model parameters
    for k in range(len(labels)):
        
        print('########################################################################')
        _,tgt = load_sample_data(file_id,k+1)
        comp = int(len(tgt)/4)
        print('Chain size: ', comp)

        for i in range(4):
            rang = range(i*comp, i*comp + comp)

            x = np.linspace(tgt[rang].min(),tgt[rang].max(),len(tgt[rang]))

            best_fit_name, best_fit_params, best_pdf = best_fit_distribution([x,tgt[rang]])

            print('Parameter', labels[k], 'is::',best_fit_name)
            print('Dist. params::',best_fit_params)
            grid, pdf = kde(tgt[rang])
            print('Local Mode: ', grid[np.argmax(pdf)], 'Local Median: ', np.median(tgt[rang]), 'Local mean: ', np.mean(tgt[rang]))
            # 
            print('Erro moda = ', abs(grid[np.argmax(pdf)] - best_fit_params[0]))
            # plot
            smp = pd.Series(tgt[rang])
        
            # plt.plot(x,best_pdf,c='k',linewidth=5)

            sns.kdeplot(smp, color='k')
            sns.histplot(smp, stat='density',palette="ch:s=.25,rot=-.25", bins=50)
            #sns.displot(smp, stat='density' ,color='b')
            
            
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.xlabel(labels[k])
            plt.ylabel('Density')
            plt.xlim((limits[k][0], limits[k][1]))
            #plt.legend(loc='best')
            # plt.tight_layout()
            name = 'pdfs/' + labels_name[k] + '_param_fit' + str(i) + '.jpeg'
    #        plt.show()
            plt.savefig(name, dpi=300)
            plt.clf()

        print('\n\n')
        x,tgt = load_sample_data(file_id,k+1)
        best_fit_name, best_fit_params, best_pdf = best_fit_distribution([x,tgt])
        print('Parameter', labels[k], 'is::',best_fit_name)
        print('Dist. params::',best_fit_params)
        grid, pdf = kde(tgt)
        print('Global Mode: ', grid[np.argmax(pdf)])
        print('Global Median: ', np.median(tgt))
        print('Global mean: ', np.mean(tgt))
        print('########################################################################')
        print('')
        

        arg   = best_fit_params[:-2]
        loc   = best_fit_params[-2]
        scale = best_fit_params[-1]

        best_dist = getattr(st, best_fit_name)
        
        pdf_fitted = best_dist.pdf(x, loc=loc, scale=scale, *arg)
        #plt.plot(x, pdf_fitted, '-',label=best_fit_name)
        
        
#        param = st.norm.fit(tgt)
#        mu , sigma = param
#        pdf_fitted = st.norm.pdf(x, *param)
#        print('Normal Fit :: ',labels[k],mu,sigma, 'COV: ',coeff_var(mu,sigma))
#        plt.plot(x,pdf_fitted,'-',label="normal fit")
        
#        param = st.lognorm.fit(tgt)
#        shape, loc, scale = param
#        mean , std = np.log(scale) , shape
#        print('Log-Normal Fit :: ',labels[k],mean,std, 'COV: ',coeff_var_ln(std))
#        pdf_fitted = st.lognorm.pdf(x, *param)
#        plt.plot(x, pdf_fitted, '-',label="logn fit")

#        (low, higher) = st.uniform.fit(tgt)
#        print('Uniform Fit :: ',labels[k],low,higher)
#        pdf_fitted = st.uniform.pdf(x, low,higher)
#        plt.plot(x, pdf_fitted, '-',label="uniform fit")

#        param = st.rayleigh.fit(tgt)
#        loc, scale = param
#        print('Rayleigh Fit :: ',labels[k],loc, scale)
#        pdf_fitted = st.rayleigh.pdf(x, *param)
#        plt.plot(x, pdf_fitted, '-',label="rayleigh fit")

        smp = pd.Series(tgt)

        # plt.plot(x,best_pdf,c='k',linewidth=5)
        sns.kdeplot(smp, color='k')
        sns.histplot(smp, stat='density',palette="ch:s=.25,rot=-.25", bins=50)
        
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlabel(labels[k])
        plt.ylabel('Density')
        plt.xlim((limits[k][0], limits[k][1]))
        #plt.legend(loc='best')
        # plt.tight_layout()
        name = 'pdfs/' + labels_name[k] + '_param_fit.jpeg'
#        plt.show()
        plt.savefig(name, dpi=300)
        plt.clf()
    
    
