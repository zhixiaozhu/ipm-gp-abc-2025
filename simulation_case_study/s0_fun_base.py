import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt


# Functions for loading data, considering size as the only covariate.
# Functions for processing data (used for model fitting and initialization).
def XY_sur_compu(popu_data):
    index_sur = np.logical_not(np.isnan(popu_data["surv"]))

    size_sur = popu_data["size"][index_sur].to_numpy()
    surv_sur = popu_data["surv"][index_sur].to_numpy()

    return (pd.DataFrame(size_sur, columns=['size']), pd.DataFrame(surv_sur, columns=['sur']))

def XY_grw_f_compu(popu_data):
    index_grw = np.logical_not(np.isnan(popu_data["sizeNext"])) & np.logical_not(np.isnan(popu_data["size"]))
    fec_grw = popu_data["fec"][index_grw]
    
    size_grw_f = popu_data["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = popu_data["sizeNext"][index_grw][fec_grw == 1].to_numpy()

    return (pd.DataFrame(size_grw_f, columns=['size']), pd.DataFrame(sizeNext_grw_f, columns=['sizeNext']))
    

def XY_grw_nf_compu(popu_data):
    index_grw = np.logical_not(np.isnan(popu_data["sizeNext"])) & np.logical_not(np.isnan(popu_data["size"]))
    fec_grw = popu_data["fec"][index_grw]
    
    size_grw_nf = popu_data["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = popu_data["sizeNext"][index_grw][fec_grw == 0].to_numpy()

    return (pd.DataFrame(size_grw_nf, columns=['size']), pd.DataFrame(sizeNext_grw_nf, columns=['sizeNext']))


def XY_grw_compu(popu_data):
    # if we are considering growth kernel for breeders and non-breeders are the SAME.
    index_grw = np.logical_not(np.isnan(popu_data["sizeNext"])) & np.logical_not(np.isnan(popu_data["size"]))  

    size_grw = popu_data["size"][index_grw].to_numpy()
    sizeNext_grw = popu_data["sizeNext"][index_grw].to_numpy()

    return (pd.DataFrame(size_grw, columns=['size']), pd.DataFrame(sizeNext_grw, columns=['sizeNext']))


def XY_fec_compu(popu_data):
    index_fec = np.logical_not(np.isnan(popu_data["fec"]))

    size_fec = popu_data["size"][index_fec].to_numpy()
    fec_fec = popu_data["fec"][index_fec].to_numpy()

    return (pd.DataFrame(size_fec, columns=['size']), pd.DataFrame(fec_fec, columns=['fec']))

def XY_flow_compu(popu_data):
    index_flow = popu_data["fec"] == 1
    size_flow = popu_data["size"][index_flow].to_numpy()
    flow_flow = popu_data["flow"][index_flow].to_numpy()

    return (pd.DataFrame(size_flow, columns=['size']), pd.DataFrame(flow_flow - 1, columns=['flow']))



