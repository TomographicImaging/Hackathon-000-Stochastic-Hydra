import sirf.STIR as pet
import numpy as np
from cil.framework import BlockDataContainer

def get_tau(K, prob):
    taus_np = []
    for (Ki,pi) in zip(K,prob):
        tau = Ki.adjoint(Ki.range_geometry().allocate(1.))
        # take care of edge of the FOV
        filter = pet.TruncateToCylinderProcessor()
        filter.apply(tau)
        backproj_np = tau.as_array()
        vmax = np.max(backproj_np[backproj_np>0])
        backproj_np[backproj_np==0] = 10 * vmax
        tau_np = 1/backproj_np
        tau.fill(tau_np)
        # apply filter second time just to be sure
        filter.apply(tau)
        tau_np = tau.as_array()
        tau_np[tau_np==0] = 1 / (10 * vmax)
        taus_np.append(pi * tau_np)
    taus = np.array(taus_np)
    tau_np = np.min(taus, axis = 0)
    tau.fill(tau_np)
    return tau

def get_sigmas(K):
    i = 0
    sigma = []
    xx = K.domain_geometry().allocate(1.)
    for Ki in K:
        tmp_np = Ki.direct(xx).as_array()
        tmp_np[tmp_np==0] = 10 * np.max(tmp_np)
        sigmai = Ki.range_geometry().allocate(0.)
        sigmai.fill(1/tmp_np)
        sigma.append(sigmai)
        i += 1
    sigmas = BlockDataContainer(*sigma)
    return sigmas
 
