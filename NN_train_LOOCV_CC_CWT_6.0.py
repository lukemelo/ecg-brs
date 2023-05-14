"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada NN training script
%
% Luke Melo
% Ashton Christy
% 01 May 2020
% 
% Version 6.0
% Edited 23 NOVEMBER 2020
% 
% Release Notes:
%   - Includes SQL fetching for X data and pinfo
%   - CWT processed ECGs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#%% Import Packages

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os, shutil, pandas as pd, numpy as np, pickle
pd.options.mode.chained_assignment = None  # default='warn'

# Import user libraries
import NN_fns_training_v1 as nn_train
  
#%% Set base working directory for calculations
wdir = os.getcwd()

saving_dir = os.path.join(r'Z:\Brugada_Project\NN_Results\Python\CWT_Results\LOOCV_Results')

# option to restart the calculation where it left off at walltime/error if needed
walltime_resume = True
n_models_save = 0#10**10  # save all models or just first few patients?

#%% Select Data Cohort and Leads to use

# Select cohort
cohort = 'Basale'
# cohort = 'Ajmaline'
# 
# Select Leads to Use
# use_leads = ['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR'] # all 12-lead
use_leads = ['I','II','III','V1','V2','V3','V4','V5','V6'] # 9-lead
# use_leads = ['I','II','III','V5','V6','aVF','aVL','aVR'] # 8-lead
# use_leads = ['V1','V2'] # V1 + V2
# use_leads = ['V1','V2','V3'] # V1 + V2 + V3
# # Individual Leads
# individual_lead_ix = 3
#        ix:   0   1     2    3    4    5    6    7    8     9    10    11
# use_leads = [['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR'][individual_lead_ix]]

#%% Load Data
remove_missing_phenotype = True
pinfo_csv = r'Z:\Brugada_Project\Python_Scripts\BrS_CC\ECG_DATA\all_patient_info_10_07_2020.csv'
X,y,pinfo = nn_train.load_ECG_cohort(cohort,use_leads,pinfo_csv,remove_missing_phenotype)

#%% Create directory to save NN models
if 'Ajm' in cohort:
    model_dir = os.path.join(saving_dir,'Ajmaline')
elif 'Bas' in cohort:
    model_dir = os.path.join(saving_dir,'Basale')  
save_models = True

if save_models:
    if len(use_leads) == 1:
        subdir = use_leads[0]
    elif len(use_leads) == 12:
        subdir = 'all'
    else:
        subdir = '-'.join(use_leads)
    save_path = os.path.join(model_dir,subdir)
    
    # Resume failed calculation?        
    if os.path.isfile(os.path.join(save_path,'checkpoint.txt')) or os.path.isfile(os.path.join(save_path,'checkpoint_old.txt')):
        if walltime_resume:
            try:
                f = open(os.path.join(save_path,'checkpoint.txt'), "rb")
                results = pickle.load(f) # unpickle results chrom 
                f.close()
            except:
                f = open(os.path.join(save_path,'checkpoint_old.txt'), "rb")
                results = pickle.load(f) # unpickle results chrom 
                f.close()
                shutil.copyfile(os.path.join(save_path,'checkpoint_old.txt'), os.path.join(save_path,'checkpoint.txt'))
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            results = list() 
    else:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        results = list()    
        
#%% Generate and Fit model
# results = list()
for holdout_index in range(len(y)):
    # start from checkpoint index
    if holdout_index >= len(results):
        # Use k-fold CV with LOOCV to train DNNs
        results.append(nn_train.LOOCV_TRAIN_FN(holdout_index,holdout_index <n_models_save,save_path,pinfo,X,y))
        # Write to checkpoint file
        if os.path.isfile(os.path.join(save_path,'checkpoint.txt')):
            shutil.copyfile(os.path.join(save_path,'checkpoint.txt'), os.path.join(save_path,'checkpoint_old.txt'))
        f = open(os.path.join(save_path,'checkpoint.txt'), "wb")
        pickle.dump(results,f)  # pickle
        f.close()
        
#%% Process results
for result_ix,result in enumerate(results):
    #result: [holdout_reg, training_stats_reg, training_stats, holdout_stats]
    if result_ix == 0:
        holdout_reg = result[0]
        training_stats_reg = result[1]
        training_stats = result[2]
        holdout_stats = result[3]
    else:
        holdout_reg += result[0]
        training_stats_reg = np.hstack((training_stats_reg,result[1]))
        training_stats += result[2]
        holdout_stats += result[3]
                   
pinfo['TN'] = training_stats[:,0]
pinfo['TP'] = training_stats[:,1]
pinfo['FN'] = training_stats[:,2]
pinfo['FP'] = training_stats[:,3]

pinfo['h_TN'] = holdout_stats[:,0]
pinfo['h_TP'] = holdout_stats[:,1]
pinfo['h_FN'] = holdout_stats[:,2]
pinfo['h_FP'] = holdout_stats[:,3]

#%% save data from calculations to excel workbooks
if save_models:
    pinfo.to_excel(os.path.join(save_path,'pinfo.xlsx'), sheet_name='sheet1', index=False)
    pd.DataFrame(holdout_reg).to_excel(os.path.join(save_path,'holdout_reg.xlsx'), sheet_name='sheet1', index=False)
    pd.DataFrame(training_stats_reg).to_excel(os.path.join(save_path,'training_stats_reg.xlsx'), sheet_name='sheet1', index=False) 
    pd.DataFrame(X).to_excel(os.path.join(save_path,'X.xlsx'), sheet_name='sheet1', index=False)  
    
#%% Delete checkpoint files
[os.remove(os.path.join(save_path,file)) for file in ['checkpoint.txt','checkpoint_old.txt'] if os.path.isfile(os.path.join(save_path,file))]    
    