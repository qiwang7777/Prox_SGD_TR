import numpy as np
def set_default_parameters(name):
    params = {}

    params['spsolver']  = name.replace(' ', '')
    params['outFreq']   = 1
    params['debug']     = False
    params['initProx']  = False
    params['t']         = 1.0
    params['safeguard'] = np.sqrt(np.finfo(float).eps)

    params['maxit']   = 200
    params['reltol']  = False
    params['gtol']    = 1e-7
    params['stol']    = 1e-12
    params['ocScale'] = params['t']

    params['eta1']     = 1e-4
    params['eta2']     = 0.75
    params['gamma1']   = 0.25
    params['gamma2']   = 2.5
    params['delta']    = 1.0
    params['deltamin'] = 1e-8
    params['deltamax'] = 100.0

    params['atol']    = 1e-5
    params['rtol']    = 1e-3
    params['spexp']   = 2
    params['maxitsp'] = 50

    params['useGCP']    = False
    params['mu1']       = 1e-4
    params['beta_dec']  = 0.1
    params['beta_inc']  = 10.0
    params['maxit_inc'] = 2

    params['lam_min'] = 1e-12
    params['lam_max'] = 1e12

    params["nonmono_M"] = 10
    return params


