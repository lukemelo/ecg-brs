# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada SQL diagnosis generating script for MORTARA
%
% Ashton Christy, Luke Melo
% 7 Jan 2020
% 
% Version 3.1
% Edited 07 October 2020
% For Motara ECG files
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Load packages
import numpy as np, ecglib_LV_3 as ecg, time, os, pandas as pd, LVPY_6 as lvPy
# from joblib import load
from scipy import signal

#%%
t0 = time.time()

            
leads = ["V1","I","II","III","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
avg_leads = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
lead_placement = 'high'
#%% Process and predict diagnosis

datadir = r'D:\Dropbox\Grantlab Brugada Dropbox\Brugada_Project\ECG_Valeria_DB\MORTARA_ECG_TEST_SUBSET_6'

h5_dir = r'Z:\Nextcloud\Brugada_Project\LabVIEW\Models'

#%% Read through folders and get filepaths for all csv files and store patient info

csv_files, csv_dirs, names, pnums, placements = list(), list(), list(), list(), list()
for root, dirs, files in os.walk(datadir):
    for file in files:
        if file.endswith('_Rhythm.csv'):
            # print(os.path.join(root, file))
            csv_files.append(file)
            csv_dirs.append(root)
            if 'female' in file.lower():
                names.append(' '.join([name_str for name_str in file.split('^_')[1].split('_Female_')[0].split('_') if name_str.isalpha()]).upper())
            else:
                names.append(' '.join([name_str for name_str in file.split('^_')[1].split('_Male_')[0].split('_') if name_str.isalpha()]).upper())
            placements.append(root.split('\\')[-1].lower())
            pnums.append(int(root.split('\\')[-2]))

# store patient info in dataframe            
pinfo = pd.DataFrame({"PatientID":pnums,\
                              "Names":names,\
                              "Lead Placement":placements,\
                              "CSV File Name":csv_files,\
                              "CSV File Path":csv_dirs})
# sort by patient number
pinfo = pinfo.sort_values(by=['PatientID','Lead Placement']).reset_index().drop(['index'],axis=1)

print('Total of %d .csv files found with mortara data' % len(pinfo['CSV File Name']))

#%% Keep only high or standard lead placement

good_ix = [ix for ix,placement in enumerate(pinfo['Lead Placement']) if placement == lead_placement]
pinfo = pinfo.iloc[good_ix].reset_index().drop(['index'],axis=1)

print('Total of %d patients with %s precordial lead placement' % (len(pinfo['CSV File Name']),lead_placement))

#%%
# Initialize variables
patients = []
brs_results = []
brs_probs = []

# mortara - OR lead index correlation vector
OR_lead_ixs = [0,1,2,6,7,8,9,10,11,5,4,3]

# Read through folders and get patient list
folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
folderlist.sort()
for ii,folder in enumerate(folderlist):
    files = [jj for jj in os.listdir(folder)]
    files.sort()

# take only pIDs larger than threshold patient from calibration dataset
pID_thresh = 2029
first_index = [ix for ix,pID in enumerate(pinfo['PatientID']) if pID > pID_thresh]    
pinfo = pinfo.iloc[first_index].reset_index().drop(['index'],axis=1)


#%%Gegin processing data
bas_scores,bas_stds,ajm_scores,ajm_stds,st_scores,st_stds,hrs,X = \
    list(),list(),list(),list(),list(),list(),list(),list()

start_ix = 9999999
# Process data for each patient
for kk,file in enumerate(pinfo['CSV File Name']):
    if kk < start_ix:
        try:
            # Open ECG trace file
            print('Now processing %d of %d: %s' % (kk+1,len(pinfo['CSV File Name']),os.path.join(pinfo['CSV File Path'][kk], file)))
            with open(os.path.join(pinfo['CSV File Path'][kk], file), 'r') as tr_file:
                tr_csv = pd.read_csv(tr_file,header=None)
                tr_csv = np.array(tr_csv.T[1:])
                # rearrznge order of leads to be consistent with OR data
                tr_csv = tr_csv[OR_lead_ixs]
                # transpose data back
                tr_csv = tr_csv.T
            
            # Upsample to 2000Hz from 1000Hz
            tr_final_all = signal.resample(tr_csv, 2*len(tr_csv))
            
            # find R-peaks using CWT
            leads = ['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']
            cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec = ecg.find_rpeaks_mortara(leads,tr_final_all,optplot=False)
            
            # calcualte the heart rate from CWT peaks
            hr = 120000/np.median(np.diff(cwt_peaks))   
            
            # chop ECG traces and generate final fusion lead
            tr_final,trace,noise_level_raw,noise_level_FFT,pct_outliers = ecg.generate_median_fusion(tr_final_all,cwt_peaks,leads)

            # make predictions using DNNs
            dnns = ['type1','bas','ajm']
            
            # only predict for baseline DNN
            (st_score,st_std) = (0,0)
            (bas_score,bas_std) = lvPy.LV_ECG_PREDICT(tr_final,h5_dir,dnns[1])
            (ajm_score,ajm_std) = (0,0)
            
            # accumulate results for ajbases
            ajbase_result = tuple([(st_score,st_std),(bas_score,bas_std),(ajm_score,ajm_std)])
        
            # export PDFs for the DNNs
            lvPy.LV_pdf_export(pinfo['CSV File Path'][kk],ajbase_result,trace,hr,'mor')
            
            # append results
            ajm_scores.append(ajm_score),ajm_stds.append(ajm_std)
            bas_scores.append(bas_score),bas_stds.append(bas_std)
            st_scores.append(st_score),st_stds.append(st_std)
            hrs.append(hr)
            X.append(trace)
        except:
            # When there's an error in the chopping, often because of noisy ECG
            hrs.append(None)
            st_scores.append(None),st_stds.append(None)
            bas_scores.append(None),bas_stds.append(None)
            ajm_scores.append(None),ajm_stds.append(None)

#%% Store prediction data in pinfo
pinfo['HR'] = hrs

pinfo['Type 1 Score'] = st_scores 
pinfo['Type 1 Stds'] = st_stds 
pinfo['Type 1 Diagnosis'] = np.round(st_scores)

pinfo['Baseline Score'] = bas_scores 
pinfo['Baseline Stds'] = bas_stds 
pinfo['Baseline Diagnosis'] = np.round(bas_scores)

pinfo['Ajmaline Score'] = ajm_scores 
pinfo['Ajmaline Stds'] = ajm_stds 
pinfo['Ajmaline Diagnosis'] = np.round(ajm_scores)

X = np.asarray(X)

#%% Save pinfo
pinfo.to_excel(os.path.join(datadir,'pinfo_mortara_DNN_predict.xlsx'), sheet_name='sheet1', index=False)
pd.DataFrame(bas_scores).to_excel(os.path.join(datadir,'holdout_reg_bas.xlsx'),sheet_name='sheet1', index=False)
pd.DataFrame(X).to_excel(os.path.join(datadir,'X.xlsx'),sheet_name='sheet1', index=False)