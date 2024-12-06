import sys
sys.path.append('../../../uqpylab_venv/run/non_newtonian_uqsa/ajuste-lasurf/anderson_run3/')
import os

import numpy as np

import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import arviz as az
import matplotlib.pyplot as plt
from math import exp, log

np.random.seed(1234)

from foam_models import *

def get_idx(fg_fit1, target_arr, tol):
    idx = indexes_of_values_with_tolerance(fg_fit1, target_arr, tol)
    # print(fg_fit1[idx], mu_app_fit1[idx]*1e3)

    idx_nonDuplicates = [idx[0]]
    if (len(target_arr) != 1):
        for i in range(1,len(idx)):
            if abs(fg_fit1[idx[i]] - fg_fit1[idx[i-1]]) > tol:
                idx_nonDuplicates.append(idx[i])

    aux = []
    for i in range(len(idx_nonDuplicates)):
        aux.append(idx_nonDuplicates[-i-1])
    
    return aux

def indexes_of_values_with_tolerance(arr, values, epsilon):
    indexes = []
    for i, val in enumerate(arr):
        for target_val in values:
            if abs(val - target_val) <= epsilon:
                indexes.append(i)
                break  # Stop checking for this value
                
    return indexes

def read_expData():
    # Loading input parameters and experimental data
    param_file = 'inputs/input_par_LASURF_backup.dat'
    data_file_QS  = 'inputs/LASURF-IL18-ZW04-test#04.dat'
    data_file_FR  = 'inputs/flow_rate_fg60_IL18_ZW04_test#4.dat', 'inputs/flow_rate_fg80_IL18_ZW04_test#4.dat'
    fg_FR_exp = 0.6, 0.8

    # Reads core parameters
    dpar = {}
    with open(param_file) as f:
        for line in f:
            (key, val) = line.split()
            dpar[str(key)] = float(val)
    core_params = np.array([dpar['swc'],dpar['sgr'],dpar['nw'],dpar['ng'],
                            dpar['kuw'],dpar['kug'],dpar['muw'],dpar['mug'],
                            dpar['u'],dpar['sigma'],dpar['phi'],dpar['kappa'],dpar['L']])

    # Reads Foam Quality Scan experimental data
    data   = np.loadtxt(data_file_QS, comments="#")
    fg_QS     = data[:,0]
    mu_app_QS = data[:,1] * 1.0e-3 # convert from cP to Pa.s
    mu_app_QS_std = data[:,2] * 1.0e-3 
    sw_QS = water_saturation(fg_QS, mu_app_QS, core_params)

    # Reads Flow Rate Scan experimental data
    mu_app_FR       = []
    mu_app_FR_std   = []
    ut_FR           = []
    sw_FR           = []
    for i in range(len(data_file_FR)):
        data   = np.loadtxt(data_file_FR[i], comments="#")
        mu_app_FR.append(data[:,0])
        ut_FR.append(data[:,1])
        mu_app_FR_std.append(data[:,2])
    
        mu_app_FR[i] = mu_app_FR[i]  * 1.0e-3 # convert from cP to Pa.s
        mu_app_FR_std[i] = mu_app_FR_std[i] * 1.0e-3
        ut_FR[i] = ut_FR[i] * 3.52778e-6      # convert from ft/day to m/s

        fg_FR = fg_FR_exp[i]*np.ones_like(ut_FR[i])
        sw_FR.append(water_saturation(fg_FR, mu_app_FR[i], core_params))
    

    # Summary of experimental data
    exp_data = {
        'fg_QS':            fg_QS,
        'mu_app_QS':        mu_app_QS,
        'mu_app_QS_std':    mu_app_QS_std,
        'sw_QS':            sw_QS,
        'ut_FR':            ut_FR,
        'mu_app_FR':        mu_app_FR,
        'mu_app_FR_std':    mu_app_FR_std,
        'sw_FR':            sw_FR,
        'fg_FR_exp':        fg_FR_exp
    }

    return core_params, exp_data

if __name__ == "__main__":
    
    # READING EXPERIMENTAL DATA 
    os.chdir('../../../uqpylab_venv/run/Non_Newtonian_UQSA_new/Non_Newtonian_UQSA/')
    core_params, exp_data = read_expData()
    os.chdir('../../../../mcmc_venv/run/anderson_mcmc/')
    fg_QS = exp_data['fg_QS']
    mu_app_QS = exp_data['mu_app_QS']
    sw_QS = exp_data['sw_QS']
    ut_QS = core_params[8]*np.ones_like(sw_QS)

    # OPTIMIZATION RESULTS
    # fmmob_opt, SF_opt, sfbet_opt, epcap_opt, fmcap_opt = 1.26531032e+03, 4.57824249e-01, 1.44292091e+02, 5.23718460e-01, 2.83666351e-07
    # fmmob_opt, SF_opt, sfbet_opt, epcap_opt, fmcap_opt = 5.95465755e+03, 4.57824250e-01, 1.44292111e+02, 5.23718488e-01, 1.47372386e-08
    fmcap_opt = 8.4e-08 # 688.7437945055274 0.45783802263044193 144.38275160971176 0.5246370864805638 8.4e-08


    # RANGES
    p0 , p1 , p2 , p3 = [10, 6300] , [core_params[0],1.0-core_params[1]] , [10, 1000] , [0,1] 

    # # MODEL SYNTHETIC DATA
    # # foam quality scan data
    # model = stars_full_class(core_params,sw_QS,ut_QS)
    # Y = model.vec_func(fmmob_opt, SF_opt, sfbet_opt, epcap_opt, fmcap_opt)[1] * (1.0 + 0.05*np.random.normal(0,1,sw_QS.shape))
    # Y *= 1000 # convert to cP

    # # flow rate scan data
    # xx_sw = np.linspace(core_params[0],1.0-core_params[1], 20000, endpoint=False)
    # for i in range(len(exp_data['ut_FR'])):
    #         for h in range(len(exp_data['ut_FR'][i])):
    #                 ut_prov = exp_data['ut_FR'][i][h] * np.ones_like(xx_sw)
                
    #                 model = stars_full_class(core_params, xx_sw, ut_prov)
    #                 fg_fit, mu_app_fit   = model.vec_func(fmmob_opt, SF_opt, sfbet_opt, epcap_opt, fmcap_opt)

    #                 aux = get_idx(fg_fit, [exp_data['fg_FR_exp'][i]], 1e-3)
    #                 muapp = 1000 * mu_app_fit[aux][0]
    #                 Y = np.concatenate((Y, np.array([muapp])))


    # EXPERIMENTAL DATA
    Y = np.concatenate((exp_data['mu_app_QS'], exp_data['mu_app_FR'][0], exp_data['mu_app_FR'][1])) * 1000
    std = np.max([np.max(exp_data['mu_app_QS_std']),
                  np.max(exp_data['mu_app_FR_std'][0]),
                  np.max(exp_data['mu_app_FR_std'][1])]) * 1000

    # In order to execute PyMC3 you must use this Theano function style.
    @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])
    def th_forward_model(param1,param2,param3,param4):
        # foam quality scan data
        model = stars_full_class_fmcapFixed(core_params,sw_QS,ut_QS,fmcap_opt)
        muapp_QS = model.vec_func(param1,param2,param3,param4)[1]  
        # flow rate scan data
        muapp_FRS = []
        for i in range(len(exp_data['ut_FR'])):
            model = stars_full_class_fmcapFixed(core_params, exp_data['sw_FR'][i], exp_data['ut_FR'][i],fmcap_opt)
            muapp_FRS.append(model.vec_func(param1,param2,param3,param4)[1])
        res = np.concatenate((muapp_QS,muapp_FRS[0],muapp_FRS[1]))
        return res


    # weigths 
    omega = 2.0

    alpha_QS = 0.5 * np.ones_like(exp_data['mu_app_QS'])
    alpha_QS[np.argmax(exp_data['mu_app_QS'])] *= omega

    alpha_FR1 = 0.1 * np.ones_like(exp_data['mu_app_FR'][0])

    alpha_FR2 = 0.4 * np.ones_like(exp_data['mu_app_FR'][1])
    alpha_FR2[0] *= omega

    alpha = np.concatenate((alpha_QS, alpha_FR1, alpha_FR2))
    

    # MCMC
    basic_model = pm.Model()
    with basic_model:
        # define priors
        param0   = pm.Uniform('fmmob',  lower = p0[0], upper = p0[1])
        param1   = pm.Uniform('SF',     lower = p1[0], upper = p1[1])
        param2   = pm.Uniform('sfbet',  lower = p2[0], upper = p2[1])
        param3   = pm.Uniform('epcap',  lower = p3[0], upper = p3[1])

        # param0 = pm.Exponential('fmmob', lam = 966.594175)
        # param1 = pm.Uniform('SF', lower = p1[0], upper = p1[1])
        # param2 = pm.Exponential('sfbet', lam = 144.292103)
        # param3 = pm.Exponential('epcap', lam = 0.523718502)

        # model
        mu_val = th_forward_model(param0,param1,param2,param3)
        mu_val *= 1000 # convert to cP

        # define erros in data
        # sigma_val = pm.HalfNormal('sigma', sigma=25.4)
        sigma_val = pm.HalfNormal('sigma', sigma=std)

        #define the log-likelihood
        Y_obs = pm.Normal('Y_obs', mu=mu_val * np.sqrt(alpha), sigma=sigma_val, observed=Y * np.sqrt(alpha))

        # instantiate sampler
        step = pm.Slice()
        
        # draw posterior samples
        trace = pm.sample(100000, step=step, tune=5000, cores=4)

    # text-based summary of the posteriors
    s = pm.summary(trace).round(2)
    print(s)

    sample_file = 'stars_mcmc.arv'
    fid = open(sample_file, "w")
    p0 , p1 , p2 , p3 = trace['fmmob'] , trace['SF'] , trace['sfbet'] , trace['epcap']
    for k in range(len(p0)):
        fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p0[k],p1[k],p2[k],p3[k]))
    fid.close()


    axes = az.plot_trace(trace)
    fig = axes.ravel()[0].figure
    fig.savefig('trace_plot_MCMC.png', dpi=300)
    print(pm.summary(trace))