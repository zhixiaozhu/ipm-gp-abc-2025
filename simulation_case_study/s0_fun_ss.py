import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

###########################
# 1 - summary stats first #
###########################

# (1) # individuals in sizeNext groups
# (2) # individuals in ageNext groups
# (3) # individuals in age-size groups

def sizeNext_group(input_data):
    bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3, 3.5, 4, float('inf')]
    labels = list(range(len(bins)-1))

    d = input_data.copy() 
    d['label_sizeNext'] = pd.cut(d['sizeNext'], bins=bins, labels=labels, right=False)

    return d



# ageNext_group: used to label age group
def ageNext_group(input_data):
    d = input_data.copy()
    d['label_ageNext'] = np.repeat(1, input_data.shape[0])
    d.loc[np.isnan(d['ageNext']), 'label_ageNext'] = np.nan
    # ageNext = 1 means newborns
    d.loc[d['ageNext'] == 1, 'label_ageNext'] = 0
    # ageNext [2, 5] young(1); [6, 10] middle(2); >10 white beard
    d.loc[d['ageNext'] > 5, 'label_ageNext'] = 2
    d.loc[d['ageNext'] > 10, 'label_ageNext'] = 3
    
    return d

def zero_fixed(result, bysize, byage=False):
    # just in case if there is zero individual in a certain group
    n_size = 12
    n_age = 4
    
    if (bysize==True) and (byage==True):
        full_index = pd.MultiIndex.from_arrays([np.repeat(range(n_age), n_size), np.tile(range(n_size), n_age)])
                                               #,names=result._index._names)
        f=pd.Series(0, index=full_index)
        return result.fillna(0).combine(f, max, fill_value=0)
        
    elif bysize == True:
        f=pd.Series(0, index=range(n_size), name=result.name)
        return result.fillna(0).combine(f, max, fill_value=0)
    
    elif byage == True:
        f=pd.Series(0, index=range(n_age), name=result.name)
        return result.fillna(0).combine(f, max, fill_value=0)


# calcualte the number of individuals in each group.
#  Individuals are grouped by their states at time t+1 (so, here are byageNext and bysizeNext).
def num_indi_group(input_data, bysizeNext, byageNext=False):
    
    if (bysizeNext == True) and (byageNext == True):
        data = ageNext_group(input_data)
        data = sizeNext_group(data)
        result = data.groupby(["label_ageNext", "label_sizeNext"]).size()
        return zero_fixed(result=result, bysize=bysizeNext, byage=byageNext)
    
    elif bysizeNext == True:
        data = sizeNext_group(input_data)
        result = data.groupby(["label_sizeNext"]).size()
        return zero_fixed(result=result, bysize=bysizeNext, byage=byageNext)
    
    elif byageNext == True:
        data = ageNext_group(input_data)
        result = data.groupby(["label_ageNext"]).size()
        return zero_fixed(result=result, bysize=bysizeNext, byage=byageNext)


# (4) prob.distribution of # individuals in sizeNext groups
# (5) prob.distribution of # individuals in ageNext groups
# (6) prob.distribution of # individuals in age-size groups

def prob_indi_group(input_data, bysizeNext, byageNext=False):
    result = num_indi_group(input_data, bysizeNext, byageNext)
    return pd.Series(result.values / result.values.sum(), index=result.index, name=result.name)


# (7) (8) (9) # flowering stalks in groups
# (10)(11)(12) # breeders in groups
# (13)(14)(15) # survivors in groups


def size_group(input_data):
    bins = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3, 3.5, 4, float('inf')]
    labels = list(range(len(bins)-1))
    d = input_data.copy() 
    d['label_size'] = pd.cut(d['size'], bins=bins, labels=labels, right=False)
    return d

# individuals in age groups, by their states at current time.
def age_group(input_data):
    d = input_data.copy()
    d['label_age'] = np.repeat(1, d.shape[0])
    d.loc[np.isnan(d['age']), 'label_age'] = np.nan
    # ageNext = 1 means newborns
    d.loc[d['age'] == 1, 'label_age'] = 0
    # ageNext 2-5 young(1); 5-10 middle(2); >10 white beard
    d.loc[d['age'] > 5, 'label_age'] = 2
    d.loc[d['age'] > 10, 'label_age'] = 3
    
    return d

# calcualte the total/mean number of breeders/flowering stlks/survivors in each group.

def num_ffs_group(input_data, target, bysize, byage=False, mean=True):
    # target should be "flow", "fec" or "surv"
    if target != 'flow' and target != 'fec' and target !=  "surv": 
        sys.exit("target should be 'flow', 'fec' or 'surv'") 
    
    # setting
    if mean == True:
        mode = 'mean'
    else:
        mode = 'sum'
        
    # calculations
    if (bysize==True) and (byage==True):
        data = age_group(input_data)
        data = size_group(data)
        result = getattr(data.groupby(["label_age", "label_size"])[target], mode)
        return zero_fixed(result=result(), bysize=bysize, byage=byage)
        
    elif bysize == True:
        data = size_group(input_data)
        result = getattr(data.groupby("label_size")[target], mode)
        return zero_fixed(result=result(), bysize=bysize, byage=byage)
        
    elif byage == True:
        data=age_group(input_data)
        result = getattr(data.groupby("label_age")[target], mode)
        return zero_fixed(result=result(), bysize=bysize, byage=byage)


# (25)(26) breeders' age/size distribution
# (27)(28) non-breeders' age/size distribution
# (29) breeders' and non-breeders' survival distribution

def fer_dis(input_data, target, bysizeNext, byageNext=False):
    if target != 'breeder' and target != 'non-breeder': 
        sys.exit("target should be 'breeder' or 'non-breeder'") 
    
    if (bysizeNext == True) and (byageNext == True):
        sys.exit("bysizeNext == True and byageNext == True") 
    
    if bysizeNext == True:
        data = sizeNext_group(input_data)
        # .sum() and .count() would drop nan if 'fec' contains nan, while .size() would not.
        result_sum = data.groupby("label_sizeNext")['fec'].sum()
        result_sum = zero_fixed(result=result_sum, bysize=bysizeNext, byage=byageNext)
        
        if target == 'breeder':
            return pd.Series(result_sum.values / result_sum.values.sum(), 
                             index=result_sum.index, name=result_sum.name)
        else:
            result_total = data.groupby("label_sizeNext")['fec'].count()
            result_total = zero_fixed(result=result_total, bysize=bysizeNext, byage=byageNext)
            result_n = result_total - result_sum
            return pd.Series(result_n.values / result_n.values.sum(), index=result_n.index, name=result_n.name)
    
    
    if byageNext == True:
        data = ageNext_group(input_data)
        result_sum = data.groupby("label_ageNext")['fec'].sum()
        result_sum = zero_fixed(result=result_sum, bysize=bysizeNext, byage=byageNext)
        
        if target == 'breeder':
            return pd.Series(result_sum.values / result_sum.values.sum(), 
                             index=result_sum.index, name=result_sum.name)
        else:
            result_total = data.groupby("label_ageNext")['fec'].count()
            result_total = zero_fixed(result=result_total, bysize=bysizeNext, byage=byageNext)
            result_n = result_total - result_sum
            return pd.Series(result_n.values / result_n.values.sum(), index=result_n.index, name=result_n.name)
        

def fer_surv_dis(input_data):
    result = input_data.groupby(["fec", 'surv']).size()
    full_index = pd.MultiIndex.from_arrays([np.repeat(range(2), 2), np.tile(range(2), 2)])
                                            #， names=result._index._names)
    f=pd.Series(0, index=full_index)
    result_fixed = result.fillna(0).combine(f, max, fill_value=0)
    return pd.Series(result_fixed.values / result_fixed.values.sum(), index=result_fixed.index, name=result_fixed.name)
    





########################
# 2 - Distance metrics #
########################

# y_exp, y_obs are the summary statstics calculated by Part 1.
# y_exp stands for the real dataset. y_obs represents the pbservations in the simulations.

# (a) Chi2 distance metric comes with the correcation to fix the values less than 5 in the expectation. 

# check and fix if the last element is less than 5
def five_fix_last(y_exp, y_obs, less_five):
    # less_five is a list of boolean: y_exp.values < 5
    if less_five[-1] == True:
        y_exp_v = np.copy(y_exp.values)
        y_obs_v = np.copy(y_obs.values)
        while (y_exp_v[-1] < 5):
            y_exp_v[-2] = y_exp_v[-2] + y_exp_v[-1]
            y_obs_v[-2] = y_obs_v[-2] + y_obs_v[-1]
            y_exp_v = np.delete(y_exp_v, -1)
            y_obs_v = np.delete(y_obs_v, -1)
            
        return (y_exp_v, y_obs_v)
        
    else:
        return (y_exp.values, y_obs.values)

def less_five_correction(y_exp, y_obs):
    less_five = y_exp.values < 5
    if np.any(less_five):
        # ensure the last element is greater or equal to 5
        (y_exp_v, y_obs_v) = five_fix_last(y_exp, y_obs, less_five)
        y_exp_v_new = np.array([0, y_exp_v[-1]])
        y_obs_v_new = np.array([0, y_obs_v[-1]])
        
        # correcting other elements that are < 5.
        for i in np.arange(y_exp_v.shape[0]-1)[::-1]:
            # np.arange(y_exp_v.shape[0]-1)[::-1]： backwords
            less_five = (y_exp_v[i] < 5)
            if less_five:
                y_exp_v_new[-1] = y_exp_v_new[-1] + y_exp_v[i]
                y_obs_v_new[-1] = y_obs_v_new[-1] + y_obs_v[i]
                
            else:
                y_exp_v_new = np.append(y_exp_v_new, y_exp_v[i])
                y_obs_v_new = np.append(y_obs_v_new, y_obs_v[i])
                
        # delete zero
        y_exp_v_new = np.delete(y_exp_v_new, 0)
        y_obs_v_new = np.delete(y_obs_v_new, 0)
        return (y_exp_v_new[::-1], y_obs_v_new[::-1])
        
    else:
        return (y_exp.values, y_obs.values)


# another way to do the correction
def less_five_correction2(y_exp, y_obs):
    less_five = y_exp.values < 5
    if np.any(less_five):
        less_five_sum = y_exp.values[less_five].sum()
        if less_five_sum >= 5:
            y_exp_fixed = np.append(y_exp.values[np.logical_not(less_five)], less_five_sum)
            y_obs_fixed = np.append(y_obs.values[np.logical_not(less_five)], 
                                    y_obs.values[less_five].sum())
            
            return (y_exp_fixed, y_obs_fixed)
        
        else:
            y_exp_fixed = y_exp.values[np.logical_not(less_five)]
            # np.argmin(y_exp_fixed) returns the first minimum.
            y_exp_fixed[np.argmin(y_exp_fixed)] = y_exp_fixed[np.argmin(y_exp_fixed)]+less_five_sum
            
            y_obs_fixed = y_obs.values[np.logical_not(less_five)]
            
            y_obs_fixed[np.argmin(y_exp_fixed)] = y_obs_fixed[np.argmin(y_exp_fixed)] + y_obs.values[less_five].sum()
            
            return (y_exp_fixed, y_obs_fixed)
        
    else:
        return (y_exp.values, y_obs.values)
        

def D_chi2(y_exp, y_obs, details=False):
    # a function to calculate the chi2 statistics. 
    # y_exp, y_obs are pd.DataFrame.Series.
    
    # Values in expectation should be greater or equal to 5.
    # y_exp_fixed, y_obs_fixed are pd.DataFrame.Series.values.
    (y_exp_fixed, y_obs_fixed) = less_five_correction(y_exp, y_obs)
    
    # Just in case....
    if np.any(y_exp_fixed < 5):
        sys.exit("Expectation is less than 5!") 
        
    chi2_array = (y_exp_fixed - y_obs_fixed) **2 / y_exp_fixed
    
    if details:
        return (chi2_array.sum(), chi2_array)
    else: 
        return chi2_array.sum()


# (b) Sum of squared differences
def D_ssd(y_exp, y_obs, details=False):
    ssd = (y_exp.values - y_obs.values) **2
    
    if details:
        return (ssd.sum(), ssd)
    else: 
        return ssd.sum()


# The remaining distance metrics are for probaility distributions.

# (c) Bhattacharyya distance 
def D_bha(p_exp, p_obs):
    return -np.log(np.sqrt(p_exp.values*p_obs.values).sum())

# (d) Hellinger distance
def D_hel(p_exp, p_obs, squared=True):
    d = np.sqrt(p_exp.values) - np.sqrt(p_obs.values)
    d_hel = np.sqrt(np.sum(d ** 2))/np.sqrt(2)
    
    if squared:
        return d_hel**2
    else:
        return d_hel

# (e) EMD 
# EMD for prob. distributions.
def D_emd_p(p_exp, p_obs):
    n = p_exp.values.size
    d = stats.wasserstein_distance(u_values=np.arange(n), u_weights=p_exp,
                                   v_values=np.arange(n), v_weights=p_obs)
    return d

# EMD for observations (calculations are implenmented based on emipirical distributions). 
def D_emd_raw(y_exp, y_obs):
    d = stats.wasserstein_distance(u_values=y_exp.dropna(),
                                   v_values=y_obs.dropna())
    return d


# (f) K-L = SUM_i log((P(i)/Q(i)))P(i)
# (g) Hilbert projective metric
# These require non-zero values in P(i) and Q(i).

# check and fix if the last element is zero
def zero_fix_last(p_exp, p_obs, equal_zero):
    if equal_zero[-1] == True:
        p_exp_v = np.copy(p_exp.values)
        p_obs_v = np.copy(p_obs.values)
        while (p_exp_v[-1] == 0) | (p_obs_v[-1] == 0):
            p_exp_v[-2] = p_exp_v[-2] + p_exp_v[-1]
            p_obs_v[-2] = p_obs_v[-2] + p_obs_v[-1]
            p_exp_v = np.delete(p_exp_v, -1)
            p_obs_v = np.delete(p_obs_v, -1)
            
        return (p_exp_v, p_obs_v)
        
    else:
        return (p_exp.values, p_obs.values)

def zero_correction(p_exp, p_obs):
    equal_zero = (p_exp.values == 0) | (p_obs.values == 0)
    if np.any(equal_zero):
        # ensure the last element is non-zero
        (p_exp_v, p_obs_v) = zero_fix_last(p_exp, p_obs, equal_zero)
        p_exp_v_new = np.array([0, p_exp_v[-1]])
        p_obs_v_new = np.array([0, p_obs_v[-1]])
        #label = p_exp_v.shape[0]
        
        for i in np.arange(p_exp_v.shape[0]-1)[::-1]:
            # p_exp_v.shape[0]-1 as we know that the last element is non-zero
            whetehr_zero = (p_exp_v[i] == 0) | (p_obs_v[i] == 0)
            if whetehr_zero:
                p_exp_v_new[-1] = p_exp_v_new[-1] + p_exp_v[i]
                p_obs_v_new[-1] = p_obs_v_new[-1] + p_obs_v[i]
                
            else:
                p_exp_v_new = np.append(p_exp_v_new, p_exp_v[i])
                p_obs_v_new = np.append(p_obs_v_new, p_obs_v[i])
                #label = np.copy(i)
                
        # delete zero
        p_exp_v_new = np.delete(p_exp_v_new, 0)
        p_obs_v_new = np.delete(p_obs_v_new, 0)
        return (p_exp_v_new[::-1], p_obs_v_new[::-1])
        
    else:
        return (p_exp.values, p_obs.values)

# !!
# Here, different as the previous, in order to use D_kl and D_hilbert,
#  users need to modify inputs manually.
#  i.e. (a, b) = zero_correction(p_exp, p_obs)
#       D_kl(a, b)

def D_kl(p_exp_0fixed, p_obs_0fixed, details=False):
    # Values in y_exp_0fixed and y_obs_0fixed should be non-zero.
    # They should be pd.series.values rather than pd.series
    if np.any((p_exp_0fixed == 0) | (p_obs_0fixed == 0)):
        sys.exit("The probabilities should be non-zero!") 
        
    kl = np.log(p_exp_0fixed/p_obs_0fixed) * p_exp_0fixed
    if details:
        return (kl.sum(), kl)
    else: 
        return kl.sum()
    
def D_hilbert(p_exp_0fixed, p_obs_0fixed):
    if np.any((p_exp_0fixed == 0) | (p_obs_0fixed == 0)):
        sys.exit("The probabilities should be non-zero!")
        
    max1 = np.max(p_exp_0fixed/p_obs_0fixed)
    max2 = np.max(p_obs_0fixed/p_exp_0fixed)
    return np.log(max1) + np.log(max2)


# define a new function to calculate a list of summary stats and then 
#  compare these by using many distance metrics.

def list_comparisons(exp_1step_data, simu_1step_data):
    # a list of summary stats at current time
    current_s = np.array(0)
    
    ## Calculating summary statsitcis first.
    
    # (1) prob.distribution of # individuals
    #                      in sizeNext groups
    summary1 = prob_indi_group(simu_1step_data, bysizeNext=True, byageNext=False)
    exp1 = prob_indi_group(exp_1step_data, bysizeNext=True, byageNext=False)

    
    # (4) # individuals 
    #         in sizeNext groups
    summary4 = num_indi_group(simu_1step_data, bysizeNext=True, byageNext=False)
    exp4 = num_indi_group(exp_1step_data, bysizeNext=True, byageNext=False)
        
    # (7) # flowering stalks
    #              in size groups
    summary7 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=False)
    exp7 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=False)

    
    # (10) # breeders 
    #              in size groups
    summary10 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=False)
    exp10 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=False)

    # (13) # survivors 
    #              in size groups
    summary13 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=False)
    exp13 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=False)

    
    # (16) the mean of # flowering stalks
    #                                            in size groups
    summary16 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=True)
    exp16 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=True)

    # (19) the mean of # breeders 
    #                                            in size groups
    summary19 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=True)
    exp19 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=True)

    # (22) the mean of # survivors 
    #                                            in size groups
    summary22 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=True)
    exp22 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=True)

    
    # (25) breeders' sizeNext  distribution
    summary25 = fer_dis(simu_1step_data, target='breeder', bysizeNext=True, byageNext=False)
    exp25 = fer_dis(exp_1step_data, target='breeder', bysizeNext=True, byageNext=False)

    # (27) non-breeders' sizeNext distribution
    summary27 = fer_dis(simu_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
    exp27 = fer_dis(exp_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
    
    # (29) breeders' and non-breeders' survival distribution
    summary29 = fer_surv_dis(simu_1step_data)
    exp29 = fer_surv_dis(exp_1step_data)
    
    for s in [1]:
        p_exp_t = locals()['exp' + str(s)]
        p_obs_t = locals()['summary' + str(s)]        
        # (c) Bhattacharyya distance 
        current_s = np.append(current_s, 
                              D_bha(p_exp=p_exp_t, p_obs=p_obs_t))
        # (d) Hellinger distance
        current_s = np.append(current_s, 
                              D_hel(p_exp=p_exp_t, p_obs=p_obs_t, squared=True))
        # (e) EMD 
        current_s = np.append(current_s, 
                              D_emd_p(p_exp=p_exp_t, p_obs=p_obs_t))
        # (f) K-L = SUM_i log((P(i)/Q(i)))P(i)
        # (g) Hilbert projective metric
        (p_exp_t_fixed, p_obs_t_fixed) = zero_correction(p_exp=p_exp_t, p_obs=p_obs_t)
        current_s = np.append(current_s, 
                              D_kl(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed, details=False))
        current_s = np.append(current_s, 
                              D_hilbert(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed))
        
    for s in [4, 7, 10, 13]:
        y_exp_t = locals()['exp' + str(s)]
        y_obs_t = locals()['summary' + str(s)]   
        
        # (a) Chi2 distance metric comes with the correcation.
        current_s = np.append(current_s, 
                              D_chi2(y_exp=y_exp_t, y_obs=y_obs_t, details=False))
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=y_exp_t, y_obs=y_obs_t, details=False))
        
    for s in [16, 19, 22]:
        y_exp_t = locals()['exp' + str(s)]
        y_obs_t = locals()['summary' + str(s)]   

        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=y_exp_t, y_obs=y_obs_t, details=False))  

            
    for s in [25, 27, 29]:
        p_exp_t = locals()['exp' + str(s)]
        p_obs_t = locals()['summary' + str(s)]        
        # (c) Bhattacharyya distance 
        current_s = np.append(current_s, 
                              D_bha(p_exp=p_exp_t, p_obs=p_obs_t))
        # (d) Hellinger distance
        current_s = np.append(current_s, 
                              D_hel(p_exp=p_exp_t, p_obs=p_obs_t, squared=True))
        # (e) EMD 
        current_s = np.append(current_s, 
                              D_emd_p(p_exp=p_exp_t, p_obs=p_obs_t))
        # (f) K-L = SUM_i log((P(i)/Q(i)))P(i)
        # (g) Hilbert projective metric
        (p_exp_t_fixed, p_obs_t_fixed) = zero_correction(p_exp=p_exp_t, p_obs=p_obs_t)
        current_s = np.append(current_s, 
                              D_kl(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed, details=False))
        current_s = np.append(current_s, 
                              D_hilbert(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed))
        
    # Lastly, for Empirical distributions' EMDs
    # for size
    current_s = np.append(current_s, 
                          D_emd_raw(y_exp=exp_1step_data['sizeNext'], y_obs=simu_1step_data['sizeNext']))        
    # for the number of flowering stalks
    current_s = np.append(current_s, 
                          D_emd_raw(y_exp=exp_1step_data['flow'], y_obs=simu_1step_data['flow']))          
        
    current_s = np.delete(current_s, 0)
    
    return current_s






def list_comparisons_interested(exp_1step_data, simu_1step_data, truedata_style):

    current_s = np.array(0)

    if truedata_style == 'glm':
        # (10) # breeders 
        #              in size groups
        summary10 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=False)
        exp10 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=False)
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                                D_ssd(y_exp=exp10, y_obs=summary10, details=False))


        # (16) the mean of # flowering stalks
        #                                            in size groups
        summary16 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=True)
        exp16 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=True)
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                                D_ssd(y_exp=exp16, y_obs=summary16, details=False))

        # (4) # individuals 
        #         in sizeNext groups
        summary4 = num_indi_group(simu_1step_data, bysizeNext=True, byageNext=False)
        exp4 = num_indi_group(exp_1step_data, bysizeNext=True, byageNext=False) 
        # (a) Chi2 distance metric comes with the correcation.
        current_s = np.append(current_s, 
                              D_chi2(y_exp=exp4, y_obs=summary4, details=False))
        
        # (27) non-breeders' sizeNext distribution
        summary27 = fer_dis(simu_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
        exp27 = fer_dis(exp_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
        # (c) Bhattacharyya distance 
        current_s = np.append(current_s, 
                              D_bha(p_exp=exp27, p_obs=summary27))
        
        # (13) # survivors 
        #              in size groups
        summary13 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=False)
        exp13 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=False)
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=exp13, y_obs=summary13, details=False))


    elif truedata_style == 'gp':

        # (10) # breeders 
        #              in size groups
        summary10 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=False)
        exp10 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=False)
        # (a) Chi2 distance metric comes with the correcation.
        current_s = np.append(current_s, 
                              D_chi2(y_exp=exp10, y_obs=summary10, details=False))  

        
        # (16) the mean of # flowering stalks
        #                                            in size groups
        summary16 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=True)
        exp16 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=True)
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                                D_ssd(y_exp=exp16, y_obs=summary16, details=False))
    
        # (25) breeders' sizeNext  distribution
        summary25 = fer_dis(simu_1step_data, target='breeder', bysizeNext=True, byageNext=False)
        exp25 = fer_dis(exp_1step_data, target='breeder', bysizeNext=True, byageNext=False)
        # (e) EMD 
        current_s = np.append(current_s, 
                              D_emd_p(p_exp=exp25, p_obs=summary25))
        
        # (29) breeders' and non-breeders' survival distribution
        summary29 = fer_surv_dis(simu_1step_data)
        exp29 = fer_surv_dis(exp_1step_data)
        # (g) Hilbert projective metric
        (p_exp_t_fixed, p_obs_t_fixed) = zero_correction(p_exp=exp29, p_obs=summary29)
        current_s = np.append(current_s, 
                              D_hilbert(p_exp_0fixed=p_exp_t_fixed, p_obs_0fixed=p_obs_t_fixed))

        # (13) # survivors 
        #              in size groups
        summary13 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=False)
        exp13 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=False)
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=exp13, y_obs=summary13, details=False))
         
         
    else: 
        assert False, '\n truedata_style should be glm or gp' 

    current_s = np.delete(current_s, 0)
    return current_s
