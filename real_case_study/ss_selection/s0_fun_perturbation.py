from s0_fun_IBM import *
from s0_fun_teststats import *



# define a new function to calculate a list of summary stats and then 
#  compare these by using many distance metrics.

def list_comparisons(exp_1step_data, simu_1step_data):
    # a list of summary stats at current time
    current_s = np.array(0)
    
    ## Calculating summary statsitcis first.
    
    # (1)(2)(3) prob.distribution of # individuals
    #                      in sizeNext groups
    summary1 = prob_indi_group(simu_1step_data, bysizeNext=True, byageNext=False)
    exp1 = prob_indi_group(exp_1step_data, bysizeNext=True, byageNext=False)
    #                      in ageNext groups
    summary2 = prob_indi_group(simu_1step_data, bysizeNext=False, byageNext=True)
    exp2 = prob_indi_group(exp_1step_data, bysizeNext=False, byageNext=True)
    #                      in age-size groups
    summary3 = prob_indi_group(simu_1step_data, bysizeNext=True, byageNext=True)
    exp3 = prob_indi_group(exp_1step_data, bysizeNext=True, byageNext=True)
    
    
    # (4)(5)(6) # individuals 
    #         in sizeNext groups
    summary4 = num_indi_group(simu_1step_data, bysizeNext=True, byageNext=False)
    exp4 = num_indi_group(exp_1step_data, bysizeNext=True, byageNext=False)
    #         in ageNext groups
    summary5 = num_indi_group(simu_1step_data, bysizeNext=False, byageNext=True)
    exp5 = num_indi_group(exp_1step_data, bysizeNext=False, byageNext=True)
    #         in age-size groups
    summary6 = num_indi_group(simu_1step_data, bysizeNext=True, byageNext=True)
    exp6 = num_indi_group(exp_1step_data, bysizeNext=True, byageNext=True)

        
    # (7)(8)(9) # flowering stalks
    #              in size groups
    summary7 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=False)
    exp7 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=False)
    #              in age groups
    summary8 = num_ffs_group(simu_1step_data, target="flow", bysize=False, byage=True, mean=False)
    exp8 = num_ffs_group(exp_1step_data, target="flow", bysize=False, byage=True, mean=False)
    #              in age-size groups
    summary9 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=True, mean=False)
    exp9 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=True, mean=False)

    
    # (10)(11)(12) # breeders 
    #              in size groups
    summary10 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=False)
    exp10 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=False)
    #              in age groups
    summary11 = num_ffs_group(simu_1step_data, target="fec", bysize=False, byage=True, mean=False)
    exp11 = num_ffs_group(exp_1step_data, target="fec", bysize=False, byage=True, mean=False)
    #              in age-size groups
    summary12 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=True, mean=False)
    exp12 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=True, mean=False)
    
    # (13)(14)(15) # survivors 
    #              in size groups
    summary13 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=False)
    exp13 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=False)
    #              in age groups
    summary14 = num_ffs_group(simu_1step_data, target="surv", bysize=False, byage=True, mean=False)
    exp14 = num_ffs_group(exp_1step_data, target="surv", bysize=False, byage=True, mean=False)
    #              in age-size groups
    summary15 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=True, mean=False)
    exp15 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=True, mean=False)
    
    
    # (16)(17)(18) the mean of # flowering stalks
    #                                            in size groups
    summary16 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=False, mean=True)
    exp16 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=False, mean=True)
    #                                            in age groups
    summary17 = num_ffs_group(simu_1step_data, target="flow", bysize=False, byage=True, mean=True)
    exp17 = num_ffs_group(exp_1step_data, target="flow", bysize=False, byage=True, mean=True)
    #                                            in age-size groups
    summary18 = num_ffs_group(simu_1step_data, target="flow", bysize=True, byage=True, mean=True)
    exp18 = num_ffs_group(exp_1step_data, target="flow", bysize=True, byage=True, mean=True)
    
    # (19)(20)(21) the mean of # breeders 
    #                                            in size groups
    summary19 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=False, mean=True)
    exp19 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=False, mean=True)
    #                                            in age groups
    summary20 = num_ffs_group(simu_1step_data, target="fec", bysize=False, byage=True, mean=True)
    exp20 = num_ffs_group(exp_1step_data, target="fec", bysize=False, byage=True, mean=True)
    #                                            in age-size groups
    summary21 = num_ffs_group(simu_1step_data, target="fec", bysize=True, byage=True, mean=True)
    exp21 = num_ffs_group(exp_1step_data, target="fec", bysize=True, byage=True, mean=True)
    
    # (22)(23)(24) the mean of # survivors 
    #                                            in size groups
    summary22 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=False, mean=True)
    exp22 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=False, mean=True)
    #                                            in age groups
    summary23 = num_ffs_group(simu_1step_data, target="surv", bysize=False, byage=True, mean=True)
    exp23 = num_ffs_group(exp_1step_data, target="surv", bysize=False, byage=True, mean=True)
    #                                            in age-size groups
    summary24 = num_ffs_group(simu_1step_data, target="surv", bysize=True, byage=True, mean=True)
    exp24 = num_ffs_group(exp_1step_data, target="surv", bysize=True, byage=True, mean=True)
    
    
    # (25)(26) breeders' sizeNext and ageNext distribution
    summary25 = fer_dis(simu_1step_data, target='breeder', bysizeNext=True, byageNext=False)
    exp25 = fer_dis(exp_1step_data, target='breeder', bysizeNext=True, byageNext=False)
    
    summary26 = fer_dis(simu_1step_data, target='breeder', bysizeNext=False, byageNext=True)
    exp26 = fer_dis(exp_1step_data, target='breeder', bysizeNext=False, byageNext=True)
    
    # (27)(28) non-breeders' sizeNext and ageNext distribution
    summary27 = fer_dis(simu_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
    exp27 = fer_dis(exp_1step_data, target='non-breeder', bysizeNext=True, byageNext=False)
    
    summary28 = fer_dis(simu_1step_data, target='non-breeder', bysizeNext=False, byageNext=True)
    exp28 = fer_dis(exp_1step_data, target='non-breeder', bysizeNext=False, byageNext=True)
    
    # (29) breeders' and non-breeders' survival distribution
    summary29 = fer_surv_dis(simu_1step_data)
    exp29 = fer_surv_dis(exp_1step_data)
    
    for s in range(1, 4):
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
        
    for s in range(4, 16):
        y_exp_t = locals()['exp' + str(s)]
        y_obs_t = locals()['summary' + str(s)]   
        
        # (a) Chi2 distance metric comes with the correcation.
        current_s = np.append(current_s, 
                              D_chi2(y_exp=y_exp_t, y_obs=y_obs_t, details=False))
        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=y_exp_t, y_obs=y_obs_t, details=False))
        
    for s in range(16, 25):
        y_exp_t = locals()['exp' + str(s)]
        y_obs_t = locals()['summary' + str(s)]   

        # (b) Sum of squared differences
        current_s = np.append(current_s, 
                              D_ssd(y_exp=y_exp_t, y_obs=y_obs_t, details=False))  

            
    for s in range(25, 30):
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
    # for age
    current_s = np.append(current_s, 
                          D_emd_raw(y_exp=exp_1step_data['ageNext'], y_obs=simu_1step_data['ageNext'])) 
    # for size
    current_s = np.append(current_s, 
                          D_emd_raw(y_exp=exp_1step_data['sizeNext'], y_obs=simu_1step_data['sizeNext']))        
    # for the number of flowering stalks
    current_s = np.append(current_s, 
                          D_emd_raw(y_exp=exp_1step_data['flow'], y_obs=simu_1step_data['flow']))          
        
    current_s = np.delete(current_s, 0)
    
    return current_s




# if only a signle hyper-parameter is to be perturbed.
def single_p_perturbation(models, perturb_model, popu_dataset, target_kernel, col_names, target_likeli=None,
                          rep_time=1000, sample_size=150, NB=False, perturb_para_index = [0, 0], 
                          perturb_level=[0.9, 0.95, 1, 1.05, 1.1]):
    # popu_dataset is a dataset contains all individuals
    target_model = models[perturb_model]
    # copy parameters from the model in to a matrix.
    if target_likeli == None:
        p_matrix = np.zeros((len(target_kernel.parameters), 2))
        for i in range(len(target_kernel.parameters)):
            p_matrix[i,:] = np.array(target_kernel.parameters[i])
    else:
        p_matrix = np.copy(np.array(target_likeli.parameters[0]))
        
        
    
    for i in range(len(perturb_level)):
        if target_likeli == None:
        
            per_p_matrix = np.copy(p_matrix)
            per_p_matrix[tuple(perturb_para_index)] *= perturb_level[i]
        
        else:
            
            per_p_matrix = np.copy(p_matrix)
            per_p_matrix *= perturb_level[i]
            
        
        globals()['summary_data' + str(i)] = pd.DataFrame(data=0.0,  index=range(rep_time), columns=col_names)
        
        for j in range(rep_time):
            
            current_dataset = popu_dataset.sample(n=sample_size)
            
            if target_likeli == None:
                models[perturb_model].assign_kernel_compiled(per_p_matrix)
                if j == 1:
                    print('After: \n')
                    print(models[perturb_model].print_kernel_compiled(0))
            else:
                models[perturb_model].assign_likeli_compiled(tf.convert_to_tensor(per_p_matrix, 
                                                                                  dtype=default_float()))
                if j == 1:
                    print('After: \n')
                    print(models[perturb_model].print_likeli_compiled(0))
                
            
            # generating dataset with perturbed parameters
            simu_1step_data = IBM_1step(zt=current_dataset["size"], age=current_dataset["age"],
                                        models=models, NB=NB)
            #print(simu_1step_data)
            
            if target_likeli == None:
                models[perturb_model].assign_kernel_compiled(p_matrix)
                if j == 1:
                    print('Original: \n')
                    print(models[perturb_model].print_kernel_compiled(0))
            else:
                models[perturb_model].assign_likeli_compiled(tf.convert_to_tensor(p_matrix, 
                                                                                  dtype=default_float())) 
                if j == 1:
                    print('Original: \n')
                    print(models[perturb_model].print_likeli_compiled(0))

            
            # generating dataset with the original parameters
            exp_1step_data = IBM_1step(zt=current_dataset["size"], age=current_dataset["age"],
                                       models=models, NB=NB)
            #print(exp_1step_data)
            
            # a list of summary stats at current time
            current_s = list_comparisons(exp_1step_data=exp_1step_data,
                                         simu_1step_data=simu_1step_data)
            globals()['summary_data' + str(i)].iloc[j,:] = current_s            

    res = (globals()['summary_data' + str(r)] for r in range(len(perturb_level)))        
    return res




# if we want to compare populations generated by two different models.
def model_perturbation(models1, models2, popu_dataset, col_names, rep_time=1000, sample_size=150, NB=False):
    # We need a dataset to recor the calculated suammary stats,
    #  there are 52 combinations in total
    
    summary_data = pd.DataFrame(data=0.0,  index=range(rep_time), columns=col_names)
    
    for j in range(rep_time):
        
        if sample_size is None:
            current_dataset = popu_dataset
        else:
            current_dataset = popu_dataset.sample(n=sample_size)
                    
        # generating dataset with models1
        simu_1step_data1 = IBM_1step(zt=current_dataset["size"], age=current_dataset["age"],
                                     models=models1, NB=NB)
        
        # generating datasetwith models2
        simu_1step_data2 = IBM_1step(zt=current_dataset["size"], age=current_dataset["age"],
                                     models=models2, NB=NB)
            
        # a list of summary stats at current time
        current_s = list_comparisons(exp_1step_data=simu_1step_data1, simu_1step_data=simu_1step_data2)
        summary_data.iloc[j,:] = current_s            
        
    
    return summary_data



# if we want to compare populations generated by the same model.
def samemodel_compare(models, input_size, input_age, popu_dataset, col_names, rep_time=1000, NB=False):
    # We need a dataset to recor the calculated suammary stats,
    #  there are 51 combinations in total
    
    summary_data = pd.DataFrame(data=0.0,  index=range(rep_time), columns=col_names)
    
    for j in range(rep_time):
        
        # generating dataset with models1
        simu_1step_data = IBM_1step(zt=input_size, age=input_age, models=models, NB=NB)
            
        # a list of summary stats at current time
        current_s = list_comparisons(exp_1step_data=popu_dataset, simu_1step_data=simu_1step_data)
        summary_data.iloc[j,:] = current_s            
        
    
    return summary_data