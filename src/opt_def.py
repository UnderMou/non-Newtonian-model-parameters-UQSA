import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from scipy.linalg import norm

from foam_models import *

def exponential_func(fmmob, epcap):
    C =  0.4713153248167298 # obtained from foam_postProc.py
    return np.exp((1/epcap) * np.log(C/fmmob))

class NNFittingSTARS(ElementwiseProblem):

    def __init__(self, core_params, exp_data, opt_params):

        self.core_params =  core_params
        self.fg_QS =        exp_data['fg_QS']
        self.mu_app_QS =    exp_data['mu_app_QS']
        self.sw_QS =        exp_data['sw_QS']
        self.ut_FR =        exp_data['ut_FR']
        self.mu_app_FR =    exp_data['mu_app_FR']
        self.sw_FR =        exp_data['sw_FR']

        self.alpha_QS = opt_params['alpha_QS']
        self.alpha_FRS1 = opt_params['alpha_FRS1']
        self.alpha_FRS2 = opt_params['alpha_FRS2']
        self.alpha_G = opt_params['alpha_G']

        p0 , p1 , p2, p3 , p4 = opt_params['p0'], opt_params['p1'], opt_params['p2'], opt_params['p3'], opt_params['p4']
        xl = np.array([p0[0], p1[0], p2[0], p3[0], p4[0]])
        xu = np.array([p0[1], p1[1], p2[1], p3[1], p4[1]])

        super().__init__(n_var=len(xl),
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

        def check_monotonicity(mu_app2):
            g = 0.0
            for i in range(len(mu_app2)):
                for j in range(1, len(mu_app2[i])):
                    if mu_app2[i][j] >= mu_app2[i][j-1]:
                        gp = (mu_app2[i][j] - mu_app2[i][j-1]) / (max(mu_app2[i]) - min(mu_app2[i]))
                        if gp > g: g = gp
            return g

        weight_trans = 2.0  # foam quality transition weight
        
        # QUALITY SCAN RESIDUAL EVALUATION
        u_QS = self.core_params[8] * np.ones_like(self.fg_QS)
        model = stars_full_class(self.core_params, self.sw_QS, u_QS)
        mu_app1 = model.vec_func(x[0] , x[1] , x[2], x[3], x[4])[1]
        residual_QS = abs(mu_app1 - self.mu_app_QS) / (max(self.mu_app_QS) - min(self.mu_app_QS))
        id_muApp_trans = np.argmax(self.mu_app_QS)
        residual_QS[id_muApp_trans] *= weight_trans 

        
        # FLOW RATE SCAN RESIDUAL EVALUATION
        residual_FR = []
        mu_app2 = []
        for i in range(len(self.ut_FR)):
            model = stars_full_class(self.core_params, self.sw_FR[i], self.ut_FR[i])
            mu_app2.append(model.vec_func(x[0] , x[1] , x[2], x[3], x[4])[1])
            residual_FR.append(np.array(abs(mu_app2[i] - self.mu_app_FR[i]) / (max(self.mu_app_FR[i]) - min(self.mu_app_FR[i]))))
            if i == 1:
                residual_FR[i][0] *= weight_trans   # also apply a weight on this point

        sum_res = self.alpha_QS*norm(np.array(residual_QS), ord=2) + \
                  self.alpha_FRS1*norm(np.array(residual_FR[0]), ord=2) + \
                  self.alpha_FRS2*norm(np.array(residual_FR[1]), ord=2)
        
        # CHECK FOR MONOTONICTY DECREASING IN mu_app 
        g = check_monotonicity(mu_app2)

        # OBJECTIVE FUNCTION
        out["F"] = [sum_res + self.alpha_G*g]


class NNFittingSTARS_dependent(ElementwiseProblem):

    def __init__(self, core_params, exp_data, opt_params):

        self.core_params =  core_params
        self.fg_QS =        exp_data['fg_QS']
        self.mu_app_QS =    exp_data['mu_app_QS']
        self.sw_QS =        exp_data['sw_QS']
        self.ut_FR =        exp_data['ut_FR']
        self.mu_app_FR =    exp_data['mu_app_FR']
        self.sw_FR =        exp_data['sw_FR']

        self.alpha_QS = opt_params['alpha_QS']
        self.alpha_FRS1 = opt_params['alpha_FRS1']
        self.alpha_FRS2 = opt_params['alpha_FRS2']
        self.alpha_G = opt_params['alpha_G']

        p0 , p1 , p2, p3 = opt_params['p0'], opt_params['p1'], opt_params['p2'], opt_params['p3']
        xl = np.array([p0[0], p1[0], p2[0], p3[0]])
        xu = np.array([p0[1], p1[1], p2[1], p3[1]])

        super().__init__(n_var=len(xl),
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

        def check_monotonicity(mu_app2):
            g = 0.0
            for i in range(len(mu_app2)):
                for j in range(1, len(mu_app2[i])):
                    if mu_app2[i][j] >= mu_app2[i][j-1]:
                        gp = (mu_app2[i][j] - mu_app2[i][j-1]) / (max(mu_app2[i]) - min(mu_app2[i]))
                        if gp > g: g = gp
            return g

        weight_trans = 2.0  # foam quality transition weight
        
        # QUALITY SCAN RESIDUAL EVALUATION
        u_QS = self.core_params[8] * np.ones_like(self.fg_QS)
        model = stars_full_class(self.core_params, self.sw_QS, u_QS)
        mu_app1 = model.vec_func(x[0] , x[1] , x[2], x[3], exponential_func(x[0], x[3]))[1]
        residual_QS = abs(mu_app1 - self.mu_app_QS) / (max(self.mu_app_QS) - min(self.mu_app_QS))
        id_muApp_trans = np.argmax(self.mu_app_QS)
        residual_QS[id_muApp_trans] *= weight_trans 

        
        # FLOW RATE SCAN RESIDUAL EVALUATION
        residual_FR = []
        mu_app2 = []
        for i in range(len(self.ut_FR)):
            model = stars_full_class(self.core_params, self.sw_FR[i], self.ut_FR[i])
            mu_app2.append(model.vec_func(x[0] , x[1] , x[2], x[3], exponential_func(x[0], x[3]))[1])
            residual_FR.append(np.array(abs(mu_app2[i] - self.mu_app_FR[i]) / (max(self.mu_app_FR[i]) - min(self.mu_app_FR[i]))))
            if i == 1:
                residual_FR[i][0] *= weight_trans   # also apply a weight on this point

        sum_res = self.alpha_QS*norm(np.array(residual_QS), ord=2) + \
                  self.alpha_FRS1*norm(np.array(residual_FR[0]), ord=2) + \
                  self.alpha_FRS2*norm(np.array(residual_FR[1]), ord=2)
        
        # CHECK FOR MONOTONICTY DECREASING IN mu_app 
        g = check_monotonicity(mu_app2)

        # OBJECTIVE FUNCTION
        out["F"] = [sum_res + self.alpha_G*g]
