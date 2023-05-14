# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada LabVIEW function library
%
% Luke Melo Ashton Christy
% 5 Dec 2020
% 
% Edited 14 Feb 2021
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def LV_ECG_PREDICT(tr_final,h5_dir,dnn):
    """
    Generates NN predictions for baseline, ST-elevation and ajmaline data
    
    Input:
        tr_final: 2000Hz 12 lead fusion for DNN predictions
        cohort: DNN to use for prediction
        
    Output:
        -
    """    
    import numpy as np,os
    
    # handle files with missing data
    if np.all(tr_final == 0):
        dnn_score,dnn_std = 0,0
    else:
        # Load trained classifiers 
        # CWT DNNs 
        h5_dir_bas = os.path.join(h5_dir,'Basale','all')
        h5_dir_ajm = os.path.join(h5_dir,'Ajmaline','all')
        h5_dir_st = os.path.join(h5_dir,'ST-Elevation','all')
        # h5_dir_bas = r'C:\Data\BrS Async VI\Models\Basale\all'
        # h5_dir_ajm = r'C:\Data\BrS Async VI\Models\Ajmaline\all'
        # h5_dir_st = r'C:\Data\BrS Async VI\Models\ST-Elevation\all'
        
        #% Use NN to make Predictions
        if dnn == 'type1':
            print('Processing Type 1 DNN:')
            dnn_scores,dnn_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_st,tr_final)
        elif dnn == 'bas':
            print('Processing Baseline DNN:')
            dnn_scores,dnn_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_bas,tr_final)
        elif dnn == 'ajm':
            print('Processing Ajmaline DNN:')
            dnn_scores,dnn_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_ajm,tr_final)
        
        dnn_score = np.nanmean(dnn_scores)
    
    return (dnn_score,dnn_std)

def LV_ECG_CHOP(datadir,ajbase):
    """
    Generates NN predictions for baseline, ST-elevation and ajmaline data
    
    Input:
        datadir: directory with the patient ECG text files
        
    Output:
        -
    """
    
    import numpy as np, time, csv, os, re, ecglib_LV_3 as ecg
    from collections import Counter
    
    #% Process and predict diagnosis
    leads = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
    
    st_score,st_std,bas_score,bas_std,ajm_score,ajm_std,hr = -1,0,-1,0,-1,0,0
    tr_final, trace = np.zeros(shape=(18000,)),np.zeros(shape=(1800,))
    # dummy loop for exceptions
    for _ in [1]:
        try:
            # Initialize variables
            patients = list()
            
            folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
            folderlist.sort()
            if folderlist[0]==datadir:
                del folderlist[0]
        
            folder_ixs = [ix for ix,pdir in enumerate(folderlist) if ajbase in pdir.lower() and not('stand' in pdir.lower())]
            if len(folder_ixs)==1:
                folderlist = [folderlist[folder_ixs[0]]]
            else:
                folderlist = folderlist[folder_ixs]
            for ii,folder in enumerate(folderlist):
                files = [jj for jj in os.listdir(folder)]
                files.sort()
                # Select only patient folders
                if ("_" in folder.split('\\')[-2] or folder.split('\\')[-2].isdigit()) and \
                        ajbase in folder.split('\\')[-1].lower():
                    patient = folder.split('\\')[-2].split('_')[0]
                    patients.append(folder.split('\\')[-2].split('_')[0])
                    if ajbase == 'bas':
                        ajbase_print_str = 'Baseline'
                    elif ajbase == 'ajm':
                        ajbase_print_str = 'Ajmaline'
                    elif ajbase == '2 min':
                        ajbase_print_str = '2 min'
                    elif ajbase == '3 min':
                        ajbase_print_str = '3 min'
                    print('Processing %s for patient %s' % (ajbase_print_str,patient))
                    # print("Processing patient "+str(patient)+"...")
                else:
                    continue
                
                # Get "best" page number - most frequent that has V1
                pages = []    
                pages = [file.split("Page ")[1].split(".")[0] for file in files if re.findall("v1.s", file.lower())]
                page_list = [file.split("Page ")[1].split(".")[0] for file in files \
                             if "page" in file.lower() and any([xx in file.split(".")[0] for xx in leads])]
                page_ix = 0
                page_found = False
                while page_ix < len(Counter(page_list).most_common()) and not(page_found):
                    if Counter(page_list).most_common()[page_ix][0] in pages:
                        page = Counter(page_list).most_common()[page_ix][0]
                        page_found = True
                    else:
                        page_ix +=1
                if not(page_found):
                    try:
                        page = pages[0]
                    except:
                        continue
                   
                sessions = [file.split(".")[1].split(" ")[1] for file in files if re.findall("v1.s", file.lower())]
                session = sessions[0]
                
                pct_outliers, noise_level_raw, noise_level_FFT = list(), list(), list()
                # Process data by parsing through ECG folder:
                # find the files that we want to grab ECG data from:
                ecg_files = list()
                
                for kk,file in enumerate(files):
                        for lead in leads:
                            if re.match(lead+"\\b", file.split(".")[0]) and len(file.split(".")[0].split(" ")) == 1:
                                if file.split(".")[1].split("Page ")[1] == page and file.split(".")[1].split(" ")[1] == session:
                                    ecg_files.append(file)
               
                with open(os.path.join(folder,file), 'r') as tr_file:
                    ECG_trace = [float(ii) for ii in tr_file.read().split('\n')[:-1]]
               
                print('Processing data in: %s' % folder)
                # find r-peaks using CWT
                [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = ecg.find_rpeaks_folder(leads,folder,ecg_files,optplot=False)
                # calcualte the heart rate
                hr = 120000/np.median(np.diff(cwt_peaks))
                
                # calculate the average heart beat for each lead using CWT peak finder
                ECG_medians = list()                        
                for kk,file in enumerate(files):
                        for lead in leads:
                            if re.match(lead+"\\b", file.split(".")[0]) and len(file.split(".")[0].split(" ")) == 1:
                                if file.split(".")[1].split("Page ")[1] == page and file.split(".")[1].split(" ")[1] == session:
                         
                                    with open(os.path.join(folder,file), 'r') as tr_file:
                                        ECG_trace = [float(ii) for ii in tr_file.read().split('\n')[:-1]]
                                    ECG_trace = ECG_trace / max(np.abs(ECG_trace))
                                
                                    # Denoise trace
                                    [noise_freqs, FFT_noisy, FFT_freq] = ecg.ECG_FFT(ECG_trace, 25)
                                    [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ecg.ECG_filt(ECG_trace, optplot=False)
                                    
                                    # Calculate noise levels (coeff of variation) from FFT
                                    noise_level_raw.append(np.std(FFT_noisy[int(len(FFT_noisy)*.75):]) / \
                                        np.mean(FFT_noisy[int(len(FFT_noisy)*.75):]))
                                    noise_level_FFT.append(np.std(FFT_clean[int(len(FFT_clean)*.75):]) / \
                                        np.mean(FFT_clean[int(len(FFT_clean)*.75):]))
                                         
                                    # Normalize trace            
                                    ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
                                
                                    # Chop and stack on R-peak
                                    if len(cwt_peaks) != 0:
                                        ECG_stack = ecg.beat_chop(ECG_trace, cwt_peaks)
                                        # Apply PCA, other trace math
                                        [_, ECG_stack_fix, pct_outlier, _, labels] = \
                                                ecg.remove_outlier_traces(ECG_stack, max_dist=5, min_samples=10, optplot=False) #.5,10
                                        pct_outliers.append(pct_outlier)
                                        ECG_median = np.median(ECG_stack_fix, axis=0)  
                                        ECG_medians.append(ECG_median)
                                   
                #% final fusion trace
                tr_final = np.reshape(np.reshape(np.asarray(ECG_medians),(12*1500,)),(12*1500,)) 
                trace = np.copy(ECG_norm_resample(tr_final))
        except:
            continue    
    return (tr_final,trace,hr)

def LV_ECG_CHOP_MORTARA(datadir,ajbase):
    """
    Chop up the mortara data in datadir
    
    Input:
        datadir: directory containing 12-lead mortara data
        
    Output:
        -
    """        # Load packages
    import numpy as np, ecglib_LV_3 as ecg, \
        os, pandas as pd
    # from joblib import load
    from scipy import signal
    
    leads = ["V1","I","II","III","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
    
    lead_placement = 'high'
    #lead_placement = 'standard'
    
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
                try:
                    pnums.append(int(root.split('\\')[-3]))
                except:
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
    
    #% Keep only high or standard lead placement
    
    good_ix = [ix for ix,placement in enumerate(pinfo['Lead Placement']) if placement == lead_placement]
    pinfo = pinfo.iloc[good_ix].reset_index().drop(['index'],axis=1)
    
    print('Total of %d patients with %s precordial lead placement' % (len(pinfo['CSV File Name']),lead_placement))
    
    # mortara - OR lead index correlation vector
    OR_lead_ixs = [0,1,2,6,7,8,9,10,11,5,4,3]
    
    # Read through folders and get patient list
    folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
    folderlist.sort()
    for ii,folder in enumerate(folderlist):
        files = [jj for jj in os.listdir(folder)]
        files.sort()
    
    # Process data for each patient
    for kk,file in enumerate(pinfo['CSV File Name']):
        # try:
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
            
        # except:
        #     continue
                
    return(tr_final,trace,hr)

def LV_ECG_CHOP_apple_watch(datadir,file):
    """
    Chop up the mortara data in datadir
    
    Input:
        datadir: directory containing apple watch data
        file: pdf file with the apple watch trace
    Output:
        tr_final: 2000Hz 12 lead fusion for DNN predictions
        trace: 200Hz trace
        hr: heart rate
    """ 
    import ecglib_LV_3 as ecg, os, numpy as np
    # Get trace from pdf file
    trace_2k,sample_rate, hr = pdf_scrape(os.path.join(datadir,file))
    
    # find R-peaks using CWT
    leads = ['I']
    cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec = ecg.find_rpeaks_AW(leads,trace_2k,optplot=False)
    
    # chop ECG traces and generate final fusion lead
    tr_final,trace,noise_level_raw,noise_level_FFT,pct_outliers = ecg.generate_median_fusion_AW(trace_2k,cwt_peaks,leads)
    
    # normalize and mean center
    # tr_final = (tr_final-tr_final.mean()) / tr_final.std()
    # trace = (trace-trace.mean()) / trace.std()
    
    # check to see if the apple watch was placed on the wrong wrist, rechop
    if np.sum(tr_final) < 0:
        print('Worn on wrong wrist, rechopping with reversed polarity')
        # invert the polarity of the ECG
        trace_2k = -trace_2k
        
        # find R-peaks using CWT
        leads = ['I']
        cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec = ecg.find_rpeaks_AW(leads,trace_2k,optplot=False)
        
        # chop ECG traces and generate final fusion lead
        tr_final,trace,noise_level_raw,noise_level_FFT,pct_outliers = ecg.generate_median_fusion_AW(trace_2k,cwt_peaks,leads)
        
    # normalize and mean center
    tr_final = (tr_final-tr_final.mean()) / tr_final.std()
    trace = (trace-trace.mean()) / trace.std()

    return(tr_final,trace,hr)

def LabVIEW_process_patient_dir(datadir,ajbase):
    """
    Generates NN predictions for baseline, ST-elevation and ajmaline data
    
    Input:
        datadir: directory with the patient ECG text files
        
    Output:
        -
    """
    
    import numpy as np, time, csv, os, re, ecglib_LV_3 as ecg
    from collections import Counter

    #% Load trained classifiers 
    
    ## CWT DNNs 
    # h5_dir_bas = r'C:\Data\BrS Async VI\Models\Basale\all'
    # h5_dir_ajm = r'C:\Data\BrS Async VI\Models\Ajmaline\all'
    # h5_dir_st = r'C:\Data\BrS Async VI\Models\ST-Elevation\all'
    
    h5_dir_bas = os.path.join(os.getcwd(),'Basale','all')
    h5_dir_ajm = os.path.join(os.getcwd(),'Ajmaline','all')
    h5_dir_st = os.path.join(os.getcwd(),'ST-Elevation','all')
    
    #% Process and predict diagnosis
    leads = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
    
    st_score,st_std,bas_score,bas_std,ajm_score,ajm_std,hr = -1,0,-1,0,-1,0,0
    trace = np.zeros(shape=(1800,))
    
    # Initialize variables
    patients = list()
    
    folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
    folderlist.sort()
    
    folder_ixs = [ix for ix,pdir in enumerate(folderlist) if ajbase in pdir.lower() and not('stand' in pdir.lower())]
    if len(folder_ixs)==1:
        folderlist = [folderlist[folder_ixs[0]]]
    else:
        folderlist = folderlist[folder_ixs]
    for ii,folder in enumerate(folderlist):
        files = [jj for jj in os.listdir(folder)]
        files.sort()
        # Select only patient folders
        if ("_" in folder.split('\\')[-2] or folder.split('\\')[-2].isdigit()) and \
                ajbase in folder.split('\\')[-1].lower():
            patient = folder.split('\\')[-2].split('_')[0]
            patients.append(folder.split('\\')[-2].split('_')[0])
            if ajbase == 'bas':
                ajbase_print_str = 'Baseline'
            else:
                ajbase_print_str = 'Ajmaline'
            print('Processing %s for patient %s' % (ajbase_print_str,patient))
            # print("Processing patient "+str(patient)+"...")
        else:
            continue
        
        # Get "best" page number - most frequent that has V1
        pages = []    
        pages = [file.split("Page ")[1].split(".")[0] for file in files if re.findall("v1.s", file.lower())]
        page_list = [file.split("Page ")[1].split(".")[0] for file in files \
                     if "page" in file.lower() and any([xx in file.split(".")[0] for xx in leads])]
        page_ix = 0
        page_found = False
        while page_ix < len(Counter(page_list).most_common()) and not(page_found):
            if Counter(page_list).most_common()[page_ix][0] in pages:
                page = Counter(page_list).most_common()[page_ix][0]
                page_found = True
            else:
                page_ix +=1
        if not(page_found):
            try:
                page = pages[0]
            except:
                continue
            
        sessions = sessions = [file.split(".")[1].split(" ")[1] for file in files if re.findall("v1.s", file.lower())]
        session = sessions[0]
        
        pct_outliers, noise_level_raw, noise_level_FFT = list(), list(), list()
        # Process data by parsing through ECG folder:
        # find the files that we want to grab ECG data from:
        ecg_files = list()
        
        for kk,file in enumerate(files):
                for lead in leads:
                    if re.match(lead+"\\b", file.split(".")[0]) and len(file.split(".")[0].split(" ")) == 1:
                        if file.split(".")[1].split("Page ")[1] == page and file.split(".")[1].split(" ")[1] == session:
                            ecg_files.append(file)
       
        print('Processing data in: %s' % folder)
        # find r-peaks using CWT
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = ecg.find_rpeaks_folder(leads,folder,ecg_files,optplot=False)
        # calcualte the heart rate
        heart_rate = 120000/np.median(np.diff(cwt_peaks))
        
        # calculate the average heart beat for each lead using CWT peak finder
        ECG_medians = list()                        
        for kk,file in enumerate(files):
                for lead in leads:
                    if re.match(lead+"\\b", file.split(".")[0]) and len(file.split(".")[0].split(" ")) == 1:
                        if file.split(".")[1].split("Page ")[1] == page and file.split(".")[1].split(" ")[1] == session:
                 
                            with open(os.path.join(folder,file), 'r') as tr_file:
                                ECG_trace = [float(ii) for ii in tr_file.read().split('\n')[:-1]]
                            ECG_trace = ECG_trace / max(np.abs(ECG_trace))
                        
                            # Denoise trace
                            [noise_freqs, FFT_noisy, FFT_freq] = ecg.ECG_FFT(ECG_trace, 25)
                            [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ecg.ECG_filt(ECG_trace, optplot=False)
                            
                            # Calculate noise levels (coeff of variation) from FFT
                            noise_level_raw.append(np.std(FFT_noisy[int(len(FFT_noisy)*.75):]) / \
                                np.mean(FFT_noisy[int(len(FFT_noisy)*.75):]))
                            noise_level_FFT.append(np.std(FFT_clean[int(len(FFT_clean)*.75):]) / \
                                np.mean(FFT_clean[int(len(FFT_clean)*.75):]))
                                 
                            # Normalize trace            
                            ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
                        
                            # Chop and stack on R-peak
                            if len(cwt_peaks) != 0:
                                ECG_stack = ecg.beat_chop(ECG_trace, cwt_peaks)
                                # Apply PCA, other trace math
                                [_, ECG_stack_fix, pct_outlier, _, labels] = \
                                        ecg.remove_outlier_traces(ECG_stack, max_dist=5, min_samples=10, optplot=False) #.5,10
                                pct_outliers.append(pct_outlier)
                                ECG_median = np.median(ECG_stack_fix, axis=0)  
                                ECG_medians.append(ECG_median)
                                
        #% final fusion trace
        tr_final = np.reshape(np.reshape(np.asarray(ECG_medians),(12*1500,)),(12*1500,))              
                        
        #% Use NN to make Predictions
        print('Processing Type 1 DNN:')
        st_results,st_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_st,tr_final)
        print('Processing Baseline DNN:')
        bas_results,bas_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_bas,tr_final)
        print('Processing Ajmaline DNN:')
        ajm_results,ajm_std = load_evaluate_h5s_individual_lead_normalize(h5_dir_ajm,tr_final)
        
        st_score = np.mean(st_results)
        bas_score = np.mean(bas_results)
        ajm_score = np.mean(ajm_results)
        
        trace = np.copy(ECG_norm_resample(tr_final))
        hr = np.copy(heart_rate)
        
        # export pdf
        LV_pdf_export(datadir,st_score,st_std,bas_score,bas_std,ajm_score,ajm_std,trace,hr,ajbase)
    
    # if writefile == True:
    #     f.close()
    return (st_score,st_std,bas_score,bas_std,ajm_score,ajm_std,trace,hr)


################################

def ECG_norm_resample(X):
    """
    normalize and resample trace for output
    """
    import numpy as np
    # lead by lead renormalized
    X_renorm = np.zeros(len(X))
    for jj in range(12):
        lead_trace = X[jj*1500:(jj+1)*1500-1]
        X_renorm[jj*1500:(jj+1)*1500-1] = (lead_trace-np.mean(lead_trace)/np.std(lead_trace))
    X = X_renorm
    
    # New sampling frequency [Hz] for ECG data downsampled from 2000Hz
    f_new = 200
    X = ECG_resample_single(X,f_new)
    
    # arrange into tuple
    # X = tuple([X[jj*150:(jj+1)*150-1] for jj in range(12)])
    return X

def load_evaluate_h5s_AW(h5_dir,X):
    """
    Evaluate models from h5 files in directory for apple watch
    """
    
    import numpy as np, os, pickle
    from keras.models import model_from_json
    from keras import backend as K
    
    # Preprocess data
    # lead by lead renormalized
    X = (X-np.mean(X)/np.std(X))
    
    # New sampling frequency [Hz] for ECG data downsampled from 2000Hz
    f_new = 200
    X = ECG_resample_single(X,f_new)
#    print(X.shape)
    
    # Load up NN files
#    print('Searching for h5 model files in %s' % h5_dir)
    h5_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
#    print('%d models found' % len(h5_files))
#    print('Loading %d h5 model files in %s' % (len(h5_files),h5_dir))
#    print(h5_files)
    json_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.json')]
    
    ir_cal_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if 'ir_cal' in f]
    brs_results = list()
    # load nn files one at a time and evaluate
    for ix in range(len(h5_files)):
        json_file = open(json_files[ix], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(h5_files[ix])
        # check for isotonic regression calibration files
        if len(ir_cal_files)>0:
            f = open(ir_cal_files[ix], "rb")
            ir = pickle.load(f) # unpickle ir cal model 
            f.close()
            result = np.asarray(ir.predict([model.predict(np.expand_dims(X,axis=0))[0][0]])).flatten()[0]
            # print(result)
            brs_results.append(result)
        else:
#        print(model.summary())
            brs_results.append(model.predict(np.expand_dims(X,axis=0))[0][0])
        # brs_results.append(model.predict(X))
        # print(brs_results[ix])
        K.clear_session()
    brs_results = np.asarray(brs_results)
    avg_result = np.mean(brs_results)
    stdev_result = np.std(brs_results)
    print('Diagnosis score of %.3f ± %.3f' % (avg_result,stdev_result))
    if avg_result >= 0.5:
        print('Positive Diagnosis')
    else:
        print('Negative Diagnosis')
    return brs_results, stdev_result

def load_evaluate_h5s_individual_lead_normalize(h5_dir,X):
    """
    Evaluate models from h5 files in directory
    """
    
    import numpy as np, os, pickle
    from keras.models import model_from_json
    from keras import backend as K
    
    # Preprocess data
    # lead by lead renormalized
    
    X_renorm = np.zeros(len(X))
    for jj in range(12):
        lead_trace = X[jj*1500:(jj+1)*1500-1]
        X_renorm[jj*1500:(jj+1)*1500-1] = (lead_trace-np.mean(lead_trace)/np.std(lead_trace))
    X = X_renorm
    
    # New sampling frequency [Hz] for ECG data downsampled from 2000Hz
    f_new = 200
    X = ECG_resample_single(X,f_new)
#    print(X.shape)
    
    # Load up NN files
#    print('Searching for h5 model files in %s' % h5_dir)
    h5_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
#    print('%d models found' % len(h5_files))
#    print('Loading %d h5 model files in %s' % (len(h5_files),h5_dir))
#    print(h5_files)
    json_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.json')]
    
    ir_cal_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if 'ir_cal' in f]
    brs_results = list()
    # load nn files one at a time and evaluate
    for ix in range(len(h5_files)):
        json_file = open(json_files[ix], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(h5_files[ix])
        # check for isotonic regression calibration files
        if len(ir_cal_files)>0:
            f = open(ir_cal_files[ix], "rb")
            ir = pickle.load(f) # unpickle ir cal model 
            f.close()
            result = np.asarray(ir.predict([model.predict(np.expand_dims(X,axis=0))[0][0]])).flatten()[0]
            # print(result)
            brs_results.append(result)
        else:
#        print(model.summary())
            brs_results.append(model.predict(np.expand_dims(X,axis=0))[0][0])
        # brs_results.append(model.predict(X))
        # print(brs_results[ix])
        K.clear_session()
    brs_results = np.asarray(brs_results)
    avg_result = np.mean(brs_results)
    stdev_result = np.std(brs_results)
    print('Diagnosis score of %.3f ± %.3f' % (avg_result,stdev_result))
    if avg_result >= 0.5:
        print('Positive Diagnosis')
    else:
        print('Negative Diagnosis')
    return brs_results, stdev_result

def load_evaluate_h5s_individual(h5_dir,X):
    """
    Evaluate models from h5 files in directory
    """
    
    import numpy as np, os
    from keras.models import model_from_json
    from keras import backend as K
    from sklearn.isotonic import IsotonicRegression
    import pickle
    # Preprocess data
    
    # Load up NN files
#    print('Searching for h5 model files in %s' % h5_dir)
    h5_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
#    print('%d models found' % len(h5_files))
#    print('Loading %d h5 model files in %s' % (len(h5_files),h5_dir))
#    print(h5_files)
    json_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if f.endswith('.json')]
    
    ir_cal_files = [os.path.join(h5_dir,f) for f in os.listdir(h5_dir) if 'ir_cal' in f]
    brs_results = list()
    # load nn files one at a time and evaluate
    for ix in range(len(h5_files)):
        json_file = open(json_files[ix], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(h5_files[ix])
        # check for isotonic regression calibration files
        if len(ir_cal_files)>0:
            f = open(ir_cal_files[ix], "rb")
            ir = pickle.load(f) # unpickle ir cal model 
            f.close()
            result = np.asarray(ir.predict([model.predict(np.expand_dims(X,axis=0))[0][0]])).flatten()[0]
            # print(result)
            brs_results.append(result)
        else:
#        print(model.summary())
            brs_results.append(model.predict(np.expand_dims(X,axis=0))[0][0])
        # brs_results.append(model.predict(X))
        # print(brs_results[ix])
        K.clear_session()
    brs_results = np.asarray(brs_results)
    avg_result = np.mean(brs_results)
    stdev_result = np.std(brs_results)
    print('Diagnosis score of %.3f ± %.3f' % (avg_result,stdev_result))
    if avg_result >= 0.5:
        print('Positive Diagnosis')
    else:
        print('Negative Diagnosis')
    return brs_results, stdev_result


########################################################################
def pdf_scrape(pdffile):
    """
    Open .PDF file from Apple Watch and convert to SVG using Inkscape
    """
    from lxml import etree
    from svg.path import parse_path #pip install svg.path
    from PyPDF2 import PdfFileReader
    import subprocess, os, numpy as np, ecglib_LV_3 as ecg
    _ = PdfFileReader(open(pdffile, "rb"))
    _ = subprocess.run(['C:\\Progra~1\\Inkscape\\Inkscape.exe',
    # _ = subprocess.run(['C:/Progra~1/Inkscape/bin/Inkscape.exe',
                '-z', 
                '-f', pdffile, 
                '-l', pdffile.split(".")[0]+".svg"])
    
    # Parse through SVG XML
    xml = etree.parse(pdffile.split(".")[0]+".svg")
    xp_lines = r"//svg:path[@style='fill:none;stroke:#cd0a20;stroke-width:1;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:10;stroke-dasharray:none;stroke-opacity:1']"    
    polylines = xml.xpath(xp_lines, namespaces={"svg": "http://www.w3.org/2000/svg"})
    tspans = xml.xpath(r"//svg:tspan", namespaces={"svg": "http://www.w3.org/2000/svg"})
    
    os.remove(pdffile.split(".")[0]+".svg")
    
    dist = 0
    # Get EKG trace
    trace = np.array([])
    for ii,poly in enumerate(polylines):
        parsed = parse_path(poly.attrib['d'])
        segment = []
        for line in parsed:
            segment.append(line.end.imag)
        dist += parsed[-1].end.real-parsed[0].start.real
        trace = np.concatenate((trace, np.array(segment)))
    trace = -trace
    
    # Get other information
    text = []
    for tspan in tspans:
        text.append(tspan.text)
    try:
        heart_rate = int(''.join([ii for ii in text if "BPM" in ii]).split(" ")[0])
    except:
        heart_rate = 60
        print("WARNING: Heart rate not found in ECG PDF")
    sample_rate = int(''.join([ii for ii in text if "Hz" in ii]).split(",")[3].split("Hz")[0])
    lead = ''.join([ii for ii in text if "Hz" in ii]).split(",")[2].split(" ")[2]   
    print(str(heart_rate)+" BPM - "+str(sample_rate)+" Hz - Lead "+lead)
    
    numskip = 6
    # Upsample to 2000Hz
    trace_2k = ecg.resample_2kHz(trace, sample_rate, 'db9')
    # Remove first and last beats
    trace_2k = trace_2k[int(numskip*2000*60/heart_rate):-int(2000*60/heart_rate)]

    return trace_2k, sample_rate, heart_rate

def ECG_resample_single(X,f_new):
    from scipy import signal
    # Resample function from 2000Hz to f_new 
    resample_length = int(X.shape[0]*f_new/2000)
#    print('Resampling ECG signal from %d Hz to %d Hz' % (2000,f_new))
    X_filt = signal.resample(X, resample_length, t=None, axis=0, window=None)
#    print('Resampling Complete')
#    plt.subplot(211)
#    plt.plot(X)
#    plt.title("Original 2000Hz")
#    plt.subplot(212)
#    plt.plot(X_filt)
#    plt.title('Resampled %dHz' % f_new)
#    plt.show()
    return X_filt

## PDF EXPORTING #########################3

def LV_pdf_export(datadir,ajbase_result,tr_final,hr,ajbase):
    """
    create pdf to export into patient directory
    """    
    import matplotlib.pyplot as plt, numpy as np, os
    if np.all(tr_final == 0):
        None
    else:
        # unpack tuple
        ((st_score,st_std),(bas_score,bas_std),(ajm_score,ajm_std)) = ajbase_result
        
        # Initialize figure
        fig=plt.figure(figsize=(9,12))
        
        # determine patient ID
        pID = datadir.split('\\')[-2]
        
        # label ECG timestamp used
        if ajbase == 'bas':
            data_label = 'Baseline ECG'
        elif ajbase == 'ajm':
            data_label = 'Ajmaline ECG'
        elif ajbase == '2 min':
            data_label = '2 min ECG'
        elif ajbase == '3 min':
            data_label = '3 min ECG'
        elif ajbase == 'mor':
            data_label = 'Mortara ECG'
        
        # Loop through leads for subplotting
        all_leads = ['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']   
        factor = 200/2000
        for lead_ix,lead in enumerate(all_leads):        
            # Plot Leads
            ax = fig.add_subplot(4,3,lead_ix+1)
            ax.plot(tr_final[int(factor*lead_ix*1500):int(factor*((lead_ix+1)*1500-1))])
            ax.set_xticklabels([])
            ax.set_yticklabels([])  
            if lead_ix == 0:
                ax.set_title('Patient ID: %s' % pID)
            elif lead_ix == 1:
                ax.set_title(data_label)
            elif lead_ix == 2:
                ax.set_title('Heart Rate: %d' % int(hr))
                
            # type 1 DNNs
            elif lead_ix == 3:
                ax.set_title('Type 1 DNN')
            elif lead_ix == 4:
                ax.set_title('Score: %.3f ± %.3f' % (st_score,st_std))
            # type 1
            elif lead_ix == 5:
                # triage diagnosis
                if st_score >= 0.5:
                    if st_score - st_std/2 > 0.5:
                        ax.set_title('Confident BrS Positive', color='crimson')
                    else:
                        # unconfidently correct
                        ax.set_title('Unconfident BrS Positive', color='yellowgreen')
                else:
                    if st_score + st_std/2 > 0.5:
                        ax.set_title('Unconfident BrS Negative', color='orange')
                    else:
                        ax.set_title('Confident BrS Negative', color='green')
                        
            # baseline DNNs
            elif lead_ix == 6:
                ax.set_title('Baseline DNN')
            elif lead_ix == 7:
                ax.set_title('Score: %.3f ± %.3f' % (bas_score,bas_std))  
            elif lead_ix == 8:
                # triage diagnosis
                if bas_score >= 0.5:
                    if bas_score - bas_std/2 > 0.5:
                        ax.set_title('Confident BrS Positive', color='crimson')
                    else:
                        ax.set_title('Unconfident BrS Positive', color='yellowgreen')
                else:
                    if bas_score + bas_std/2 > 0.5:
                        ax.set_title('Unconfident BrS Negative', color='orange')
                    else:
                        ax.set_title('Confident BrS Negative', color='green')
    
            # ajmaline DNNs
            elif lead_ix == 9:
                ax.set_title('Ajmaline DNN')
            elif lead_ix == 10:
                ax.set_title('Score: %.3f ± %.3f' % (ajm_score,ajm_std))
            elif lead_ix == 11:
                # triage diagnosis
                if ajm_score >= 0.5:
                    if ajm_score - ajm_std/2 > 0.5:
                        ax.set_title('Confident BrS Positive', color='crimson')
                    else:
                        ax.set_title('Unconfident BrS Positive', color='yellowgreen')
                else:
                    if ajm_score + ajm_std/2 > 0.5:
                        ax.set_title('Unconfident BrS Negative', color='orange')
                    else:
                        ax.set_title('Confident BrS Negative', color='green')
     
    
    
    
            lim = np.max(np.asarray([-np.min(tr_final),np.max(tr_final)]))
            plt.ylim(-lim,lim)
            plt.legend([lead])
                
        # save the plot
        if True:#save_PDF:
            plt.savefig(os.path.join(datadir,'%s-%s.pdf' % (pID,data_label)))

def LV_pdf_export_all_cohort_ajbase(datadir,results):
    """
    create pdf to export into patient directory
    """   
    import matplotlib.pyplot as plt, numpy as np, os, pandas as pd
    
    # determine patient ID
    pID = datadir.split('\\')[-1]
    
    # define constants
    times = [0,2,3,5]
    ajbases = ['bas','2 min','3 min','ajm']
    dnns = ['type1','bas','ajm']

    # tabulate and save data
    # df_cols = ['DNN','bas score','bas std','2 min score','2 min std','3 min score','3 min std','ajm score','ajm std']
    df_cols = ['DNN'] + ajbases
    df = pd.DataFrame(columns=df_cols)
    df['DNN'] = dnns
    for dnn_ix,dnn in enumerate(dnns):
        for ajbase_ix,ajbase in enumerate(ajbases):
            # df['%s score' % ajbase][dnn_ix] = results[ajbase_ix][dnn_ix][0]
            # df['%s std' % ajbase][dnn_ix] = results[ajbase_ix][dnn_ix][1]
            df[ajbase][dnn_ix] = '%.3f ± %.3f' % (results[ajbase_ix][dnn_ix][0],results[ajbase_ix][dnn_ix][1])
    # save dataframe to excel
    df.to_csv(os.path.join(datadir,'%s-DNN-Results.csv' % pID), index=False)
    
    fig=plt.figure(figsize=(6,12))
    plt.subplot(212)
    # hide axes
    fig.patch.set_visible(False)
    plt.axis('off')
    plt.axis('tight')
    plt.table(cellText=df.values, colLabels=df.columns, loc = 'top', cellLoc='center')
    fig.tight_layout()
    
    
    # remove missing data
    real_ixs = [ix for ix,result in enumerate(results) if not(result == ((0, 0), (0, 0), (0, 0)))]
    ajbases = np.asarray(ajbases)[real_ixs]
    results = np.asarray(results)[real_ixs]
    times = np.asarray(times)[real_ixs]
    
    # Initialize figure
    # fig=plt.figure(figsize=(6,6))
    plt.subplot(211)
    for dnn_ix,dnn in enumerate(dnns):
        dnn_scores,dnn_stds = list(),list()
        for ajbase_ix,ajbase in enumerate(ajbases):
            dnn_scores.append(results[ajbase_ix][dnn_ix][0])
            dnn_stds.append(results[ajbase_ix][dnn_ix][1])
        plt.errorbar(times, dnn_scores, yerr=dnn_stds, fmt='--o')
    
    # plot cutoff value of 0.5
    plt.errorbar([0,5],[0.5,0.5],yerr=[0,0], fmt='--')
    
        
    plt.legend(['Type 1 DNN','Baseline DNN', 'Ajmaline DNN','Cutoff'])
    plt.ylabel('DNN Score')
    plt.xlabel('Time After Ajmaline')
    plt.title('ECG DNN Analysis for Patient ID: %s' % pID)
    plt.ylim(-0.1,1.1)
    plt.xlim(-0.5,5.5)
    plt.xticks([0,2,3,5], ('Baseline', '2 min', '3 min', 'Ajmaline (5 min)'))
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha = 0.2)
    # save plot
    
    fig.tight_layout()
    
    plt.savefig(os.path.join(datadir,'%s-DNN-Results.pdf' % pID),bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    
