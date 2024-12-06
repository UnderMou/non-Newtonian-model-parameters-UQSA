import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from scipy.linalg import norm

from foam_models import *
from opt_def import *

import pickle

from copy import deepcopy

import multiprocessing

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

def opt_approaches(core_params):
    approaches = []

    #Approach 1 : Newtonian approach - Fits F_dry parameters fmmob, SF, sfbet using only foam quality scan experimental data
    opt_params = {
        'alpha_QS':         1.0,
        'alpha_FRS1':       0.0,
        'alpha_FRS2':       0.0,
        'alpha_G':          0.0,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [0.0, 0.0],
        'p4':               [1.0, 1.0]
    }
    approaches.append(opt_params)

     # Approach 2.1 : Fits all parameters (fmmob, SF, sfbet, epcap and fmcap) using only foam quality scan experimental data      
    opt_params = {
        'alpha_QS':         1.0,
        'alpha_FRS1':       0.0,
        'alpha_FRS2':       0.0,
        'alpha_G':          0.0,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [0, 1],
        # 'p4':               [1.0e-08, 1.0e-04]
        'p4':               [1.0e-08, 2.09954150e-06]
    }
    approaches.append(opt_params)  

    # Approach 2.2 : Fits parameters fmmob, SF, sfbet and epcap, but keeping fmcap fixed with a physical knowledge
    #                using only foam quality scan experimental data
    fmcap_fixed = 8.4e-8     # Nca = (1.05e-3 * 2.4e-6)/0.03 menor esperado
    opt_params = {
        'alpha_QS':         1.0,
        'alpha_FRS1':       0.0,
        'alpha_FRS2':       0.0,
        'alpha_G':          0.0,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [0, 1],
        'p4':               [fmcap_fixed, fmcap_fixed]
    }
    approaches.append(opt_params)

     # Approach 3 : Fits all parameters (fmmob, SF, sfbet, epcap and fmcap) using both foam quality scan and 
    #              flow rate scan experimental data
    opt_params = {
        'alpha_QS':         0.5,
        'alpha_FRS1':       0.1,
        'alpha_FRS2':       0.4,
        'alpha_G':          0.1,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [0, 1],
        'p4':               [1.0e-08, 1.0e-04]
    }
    approaches.append(opt_params)   

    # Approach 3.2 : Fits parameters fmmob, SF, sfbet and epcap, but keeping fmcap fixed with a physical knowledge
    #                using both foam quality scan and flow rate scan experimental data
    # fmcap_fixed = 6.601449092734786e-07   -> approximated: 9.66594175e+02 4.57824250e-01 1.44292103e+02 5.23718502e-01 4.74373487e-07
    fmcap_fixed = 8.4e-8     # Nca = (1.05e-3 * 2.4e-6)/0.03 menor esperado

    opt_params = {
        'alpha_QS':         0.5,
        'alpha_FRS1':       0.1,
        'alpha_FRS2':       0.4,
        'alpha_G':          0.1,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [0, 1],
        'p4':               [fmcap_fixed, fmcap_fixed]
    }
    approaches.append(opt_params)  

    # Approach 4 : fmmob and fmcap dependency
    opt_params = {
        'alpha_QS':         0.5,
        'alpha_FRS1':       0.1,
        'alpha_FRS2':       0.4,
        'alpha_G':          0.1,
        'p0':               [1.0e+01,1.0e+04],
        'p1':               [core_params[0],1.0-core_params[1]],
        'p2':               [1.0e+01,4.0e+03],
        'p3':               [1e-4, 1]
    }
    approaches.append(opt_params) 

    return approaches

def run_optimization(seed, opt_setup):
        path = './post_proc/parameter_optimization/approach1/result_seed_' + str(seed) + '.pkl'
        print(path)

        opt_run = deepcopy(opt_setup)

        res = minimize(opt_run.problem,
                opt_run.algorithm,
                opt_run.termination,
                seed=seed,
                verbose=False)
        
        print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

        with open(path, 'wb') as outp: 
            pickle.dump(res, outp, pickle.HIGHEST_PROTOCOL)

def process_item(item):
    seed, opt_setup = item[0], item[1]
    
    run_optimization(seed, opt_setup)

class optSetup():

    def __init__(self, problem, algorithm, termination):
        self.problem = problem
        self.algorithm = algorithm
        self.termination = termination


if __name__ == '__main__':

    ################################################
    # READING EXPERIMENTAL DATA
    ################################################
    core_params, exp_data = read_expData()
    print(core_params, exp_data)
    Nca = (core_params[6] * np.concatenate((exp_data['ut_FR'][0],exp_data['ut_FR'][1])))/0.03
    print(Nca)
    teste
    

    ################################################
    # OPTIMIZATION
    ################################################
    """
    approach = 0   : approach 1
    approach = 1   : approach 2.1
    approach = 2   : approach 2.2
    approach = 3   : approach 3
    approach = 4   : approach 3.2
    approach = 5   : approach 4
    """
    approach = 0
    approaches = opt_approaches(core_params)

    NNFittingSTARS_problem = NNFittingSTARS(core_params, exp_data, approaches[approach])

    # OPTIMIZATION ALGORITHM
    # Sets the evolutionary algorithm
    algorithm = DE(
        pop_size=200,
        sampling= FloatRandomSampling(),
        variant="DE/best/1/bin",
        CR=0.6,
        F=0.7,
        dither="vector",
        jitter=False
    )
    # Sets the stop criterion
    termination = DefaultSingleObjectiveTermination(
                        n_max_gen=2000,
                        n_max_evals=400000,
                        period=300)
    
    opt_setup = optSetup(NNFittingSTARS_problem, algorithm, termination)

    n_runs = 2000   # number of independent runs of the optimization algorithm

    # Run optimizations
    root_seed = 1   # Sets seed for reproductibility of the results
    seeds = np.arange(root_seed, n_runs+1, dtype=int)
    data = [[seeds[i], opt_setup] for i in range(len(seeds))]

    num_workers = 6

    with multiprocessing.Pool(processes=num_workers) as pool:
    
        results = pool.map(process_item, data)

    # for i in range(n_runs):
    #     seed = root_seed + i
    #     print(f"Seed = {seed}")
    #     path = './post_proc/parameter_optimization/approach3/result_seed_' + str(seed) + '.pkl'
    #     print(path)

    #     res = minimize(NNFittingSTARS_problem,
    #             algorithm,
    #             termination,
    #             seed=seed,
    #             verbose=True)

    #     print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    
    #     theta = res.X

    #     results.append(res)

    #     with open(path, 'wb') as outp: 
    #         pickle.dump(res, outp, pickle.HIGHEST_PROTOCOL)