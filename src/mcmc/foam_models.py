import numpy as np

def water_saturation(fg,mu_app,mprop):
    """
    Complementary function to evaluate correspondent water saturation for a given experimental fg and mu_app
    """
    swc = mprop[0]
    sgr = mprop[1]
    nw  = mprop[2]
    kuw = mprop[4]
    muw = mprop[6]
    
    x_sw = np.zeros(len(fg))
    for k in range(len(fg)):
        if(mu_app[k] > 1.0e-08):
            x_sw[k] = swc + (1.0 - swc - sgr) * pow( (muw * (1.0 - fg[k])) / (kuw * mu_app[k]) ,1.0/nw)
    return x_sw

class stars_full_class():
    """
    This class features the CMG-STARS model
    mrf = 1.0 + fmmob * F_dry * F_shear
    """
    def __init__(self, core_params, sw, ut):
        self.swc, self.sgr = core_params[0], core_params[1]
        self.nw,  self.ng  = core_params[2], core_params[3]
        self.kuw, self.kug = core_params[4], core_params[5]
        self.muw, self.mug = core_params[6], core_params[7]
        self.u,   self.sigma_wg = ut, core_params[9]
        self.sw = sw

    def frac_flow(self):
        """
        This function returns the fractional flow theory functions
        """
        se = (self.sw - self.swc) / (1.0 - self.swc - self.sgr)
        krw = self.kuw * np.power(se,self.nw)
        krg = self.kug * np.power(1.0-se,self.ng)
        mob_w = krw / self.muw
        mob_g = krg / self.mug
        
        return krw,krg,mob_w,mob_g
        
    def vec_func(self,fmmob,SF,sfbet,epcap,fmcap):
        """
        This functions evaluates mu_app as a function of CMG-STARS parameters: fmmob, SF, sfbet, epcap, fmcap
        """

        krw,krg,mob_w,mob_g = self.frac_flow()
        
        # Evaluates F_dry term 
        Fdry = 0.5 + (1.0/np.pi) * np.arctan( sfbet * (self.sw - SF))

        # Evaluates F_shear term
        Nca = (self.muw / self.sigma_wg) * self.u

        Fshear  = np.zeros((len(Nca)))
        for i in range(len(Nca)):
            if(Nca[i] < fmcap):
                Fshear[i] = 1.0
            else:
                Fshear[i] = np.power(fmcap/Nca[i],epcap)

        # Compute mrf and results of fg, mu_app, and lt
        mrf    = 1.0 + fmmob * np.multiply(Fdry, Fshear)
        lt     = mob_w + np.divide(mob_g, mrf)
        fg     = np.divide(mob_g, mrf) / lt
        mu_app = np.divide(1.0, lt)

        # print('Sw: ', self.sw)
        # print('Fdry: ', Fdry)
        # print('Fshear: ', Fshear)
        # print('mu_app: ', mu_app)

        return fg, mu_app


class stars_class():
    """
    This class features the CMG-STARS model
    mrf = 1.0 + fmmob * F_dry
    """
    def __init__(self, core_params, sw, ut):
        self.swc, self.sgr = core_params[0], core_params[1]
        self.nw,  self.ng  = core_params[2], core_params[3]
        self.kuw, self.kug = core_params[4], core_params[5]
        self.muw, self.mug = core_params[6], core_params[7]
        self.u,   self.sigma_wg = ut, core_params[9]
        self.sw = sw

    def frac_flow(self):
        """
        This function returns the fractional flow theory functions
        """
        se = (self.sw - self.swc) / (1.0 - self.swc - self.sgr)
        krw = self.kuw * np.power(se,self.nw)
        krg = self.kug * np.power(1.0-se,self.ng)
        mob_w = krw / self.muw
        mob_g = krg / self.mug
        
        return krw,krg,mob_w,mob_g
        
    def vec_func(self,fmmob,SF,sfbet):

        krw,krg,mob_w,mob_g = self.frac_flow()
        
        # Evaluates F_dry term 
        Fdry = 0.5 + (1.0/np.pi) * np.arctan( sfbet * (self.sw - SF))

        # Compute mrf and results of fg, mu_app, and lt
        mrf    = 1.0 + fmmob * Fdry
        lt     = mob_w + np.divide(mob_g, mrf)
        fg     = np.divide(mob_g, mrf) / lt
        mu_app = np.divide(1.0, lt)
        
        return fg, mu_app
    

class stars_full_class_fmcapFixed():
    """
    This class features the CMG-STARS model
    mrf = 1.0 + fmmob * F_dry * F_shear
    """
    def __init__(self, core_params, sw, ut, fmcap):
        self.swc, self.sgr = core_params[0], core_params[1]
        self.nw,  self.ng  = core_params[2], core_params[3]
        self.kuw, self.kug = core_params[4], core_params[5]
        self.muw, self.mug = core_params[6], core_params[7]
        self.u,   self.sigma_wg = ut, core_params[9]
        self.sw = sw

        self.fmcap = fmcap

    def frac_flow(self):
        """
        This function returns the fractional flow theory functions
        """
        se = (self.sw - self.swc) / (1.0 - self.swc - self.sgr)
        krw = self.kuw * np.power(se,self.nw)
        krg = self.kug * np.power(1.0-se,self.ng)
        mob_w = krw / self.muw
        mob_g = krg / self.mug
        
        return krw,krg,mob_w,mob_g
        
    def vec_func(self,fmmob,SF,sfbet,epcap):
        """
        This functions evaluates mu_app as a function of CMG-STARS parameters: fmmob, SF, sfbet, epcap, fmcap
        """

        krw,krg,mob_w,mob_g = self.frac_flow()
        
        # Evaluates F_dry term 
        Fdry = 0.5 + (1.0/np.pi) * np.arctan( sfbet * (self.sw - SF))

        # Evaluates F_shear term
        Nca = (self.muw / self.sigma_wg) * self.u

        Fshear  = np.zeros((len(Nca)))
        for i in range(len(Nca)):
            if(Nca[i] < self.fmcap):
                Fshear[i] = 1.0
            else:
                Fshear[i] = np.power(self.fmcap/Nca[i],epcap)

        # Compute mrf and results of fg, mu_app, and lt
        mrf    = 1.0 + fmmob * np.multiply(Fdry, Fshear)
        lt     = mob_w + np.divide(mob_g, mrf)
        fg     = np.divide(mob_g, mrf) / lt
        mu_app = np.divide(1.0, lt)

        # print('Sw: ', self.sw)
        # print('Fdry: ', Fdry)
        # print('Fshear: ', Fshear)
        # print('mu_app: ', mu_app)

        return fg, mu_app
    
