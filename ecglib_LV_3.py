# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% ECG Processing Library
% Python 3.7
% 
% Ashton Christy and Luke Melo
% 20 Feb 2020
% 
% Edited:
% 32 Oct 2020
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import numpy as np, mysql.connector, math, scipy.signal as sig, \
    wx, matplotlib.pyplot as plt, os
from scipy import stats
#from modwt import modwt, imodwt

########################################
# File I/O
########################################

def phenotype(name):
    """
    Differentiate between ECG phenotypes
    
    Inputs: 
        name-- String (norm/rbbb/s1)

    Outputs:
        sn-- Phenotype string (normalized)
        pheno-- Phenotype category number (1-3)
    """
    if "norm" in name.lower():
        pheno = 1
        sn = "Normal"
#    if "irbbb" in name:
#        pheno = 3
#        sn = "IRBBB"
    elif "rbbb" in name.lower():
        pheno = 2
        sn = "(I)RBBB"
    elif "s1" in name.lower():
        pheno = 3
        sn = "Spontaneous Type 1"
    else:
        print("Phenotype "+name+" not recognized!  Assuming normal...")
        pheno = 1
        sn = "Normal (?)"
    return sn, pheno

def binarize_sex(sex):
    """
    Convert patient sex to categorical
    
    Input:
        sex-- String (m/f)
    
    Output:
        out-- Binary value
    """
    if "m" in sex.lower().strip():
        out = 1
    elif "f" in sex.lower().strip():
        out = 0
    else:
        out = 2
    return out

def get_path(setpath):
    """
    Dialog window for folder selection
    
    Input:
        setpath-- Starting path
        
    Output:
        path-- User-selected path
    """
    _ = wx.App(None)
    style = wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
    dialog = wx.DirDialog(None, "Select folder", "", style=style)
    dialog.SetPath(setpath)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def get_file(wildcard, setpath):
    """
    Dialog window for file selection 
    
    Inputs:
        wildcard-- Filetype wildcard
        setpath-- Starting path
        
    Output:
        path-- User-selected path
    """
    _ = wx.App(None)
    style = wx.FD_DEFAULT_STYLE | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, message="Select file", \
                           defaultDir=setpath, defaultFile="", style=style)
    dialog.SetWildcard(wildcard)
    if dialog.ShowModal() == wx.ID_OK:
        file = dialog.GetPath()
    else:
        file = None
    dialog.Destroy()
    return file

def convertToBinaryData(filename):
    """
    Convert .TXT file to binary data
    
    Input:
        filename-- Full .TXT filename to read in
        
    Output:
        binaryData-- Binarized text data
    """
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def write_file(data, filename):
    """
    Write binary data to .TXT
    
    Inputs:
        data-- Data to write
        filename-- Full .TXT filename to write
        
    Outputs:
        none
    """
    with open(filename, 'wb') as file:
        file.write(data)

def insertBLOB(session, cursor, patient, ajbase, lead, fn):
    """
    Insert .TXT file in MySQL table as BLOB data type
    
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        patient-- Patient number to insert
        lead-- Lead (string) to insert
        fn-- Full .TXT filename to insert
        
    Outputs:
        none
    """
    try:
        if "ajm" in ajbase.lower():
            sql_insert_blob_query = """ UPDATE `ajmaline_traces`
                        SET `"""+lead+"""` = (%s)
                        WHERE `PatientID` = """+str(patient)+""" """
        elif "bas" in ajbase.lower():
            sql_insert_blob_query = """ UPDATE `baseline_traces`
                        SET `"""+lead+"""` = (%s)
                        WHERE `PatientID` = """+str(patient)+""" """
        file = convertToBinaryData(fn)
        insert_blob_tuple = (file)
        cursor.execute(sql_insert_blob_query, (insert_blob_tuple,))
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Failed inserting BLOB data into MySQL table {}".format(error))

def readBLOB(session, cursor, patient, cut, ajbase, lead):
    """
    Read BLOB file from MySQL table    
    
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        patient-- Patient number to insert
        cut-- Numeric flag to indicate which table:
            0--- Raw trace
            1--- Pre-PCA beat stack
            2--- Post-PCA filtered beat stack (no outliers)
            3--- Median (representative) beat
        ajbase-- String flag (ajm/bas) to select database
        lead-- Lead (string) to insert
        
    Output:
        record-- Data
    """
    try:
        if cut == 1:
            cut_tbl = "_traces_prePCA"
        elif cut == 2:
            cut_tbl = "_traces_chopped"
        elif cut == 3:
            cut_tbl = "_traces_representative"
        else:
            cut_tbl = "_traces"
        if "bas" in ajbase.lower():
            sql_fetch_blob_query = """SELECT `"""+lead+"""` FROM `baseline"""+cut_tbl+"""` WHERE PatientID = %s"""
        elif "ajm" in ajbase.lower():
            sql_fetch_blob_query = """SELECT `"""+lead+"""` FROM `ajmaline"""+cut_tbl+"""` WHERE PatientID = %s"""
        cursor.execute(sql_fetch_blob_query, (patient, ))
        record = cursor.fetchall()
        return record
    except mysql.connector.Error as error:
        session.rollback()
        print("Failed to read BLOB data from MySQL table {}".format(error))

def PrintException():
    """
    Print Python exception
    """
    import linecache, sys
    _,exc_obj,tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

#%%
########################################
# Trace math: de-noising
########################################

def ECG_FFT(ECG_trace, prom_factor):
    """
    Extract fixed-frequency noise from a trace using FFT
        
    Inputs:
        ECG_trace-- Linear ECG trace data (array)
        prom_factor-- Peak prominence used in scipy.signal.find_peaks
        
    Outputs:
        freqs-- Detected noise frequencies
        P1-- 1-sided power spectrum
        f-- Frequency vector
    """
    # Perform FFT
    Y = np.fft.fft(ECG_trace)
    # Calculate 2- and 1-sided power spectra and shift to 0 Hz
    P2 = abs(Y/len(ECG_trace))
    P1 = [P2[ii] for ii in np.arange(0,math.floor(len(ECG_trace)/2 + 1))]
    P1[1:-1] = [2*ii for ii in P1[1:-1]]
    # Calibrate frequency vector, find 50 Hz index
    f = 2000 * np.arange(0,len(ECG_trace)/2) / len(ECG_trace)
    f_idx = np.argmin(abs(f-50))
    # Find peaks in 2-sided FFT
    [peaks,_] = sig.find_peaks(P1[f_idx:], prominence=(max(P1)/prom_factor), distance=500)
    freqs = [ii+50 for ii in [round(f[int(jj)]) for jj in peaks]]
    [peaks2,_] = sig.find_peaks(P1[f_idx:], width=1, prominence = 0.001, distance=500) ### NEW
    freqs.extend([ii+50 for ii in [round(f[int(jj)]) for jj in peaks2]]) ### NEW
    # Add overtones
    freqs.extend([ii*2 for ii in freqs if ii*2 < 1000]) ### NEW
    freqs.extend([ii*3 for ii in freqs if ii*3 < 1000]) ### NEW
    # Add known peaks for 50 Hz AC hum (and overtones)
    freqs.extend(ii for ii in np.linspace(50,1000,20))
    freqs = list(set(freqs))
    freqs.sort(key=int)
    del freqs[-1] # Have to remove w=1000 Hz due to conflict with Fs = 2000 Hz NEW
    if freqs[-1] == 999: ### NEW
        del freqs[-1]
    return freqs, P1, f

def ECG_filt(ECG_trace, optplot): # Edited
    """
    Denoise trace with IIR filters
    
    Inputs:
        ECG_trace-- Linear ECG trace data (array)
        optplot-- Boolean flag to display plots
        
    Outputs:
        ECG_trace_filt_final-- Denoised linear ECG trace data (array)
        noise_freqs-- Detected noise frequencies
        FFT_noisy-- Raw power spectrum
        FFT_clean-- Denoised power spectrum
    """
    # Calculate noise frequencies
    [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 100)
    # Filter out noise
    ECG_trace_filt = ECG_trace
    for freq in noise_freqs:
        sos = []
        try:
            sos = sig.butter(9, Wn=[freq-1.5, freq+1.5], btype='stop', fs=2000, output='sos')
        except:
            print("Filter creation failure at f="+str(freq) + \
                  " Hz - Must be less than 1000 Hz.\n List of frequencies:")
            print(noise_freqs)
        ECG_trace_filt = sig.sosfiltfilt(sos, ECG_trace_filt)
    sos = sig.butter(20, Wn=[49, 51], btype='stop', fs=2000, output='sos')
    ECG_trace_filt_final = sig.sosfiltfilt(sos, ECG_trace_filt)
    # Filter out background
    sos = sig.butter(9, Wn=0.5, btype='high', fs=2000, output='sos')
    ECG_trace_filt_final = sig.sosfiltfilt(sos, ECG_trace_filt_final)
    [_,FFT_clean,_] = ECG_FFT(ECG_trace_filt_final, 25)
    if optplot:
        _ = plt.figure(figsize=(12,3))   
        try: # Even-length traces
            plt.plot(FFT_freq+50, FFT_noisy)
            plt.plot(FFT_freq+50, FFT_clean, 'r')                
        except: # Odd-length traces
            plt.plot(FFT_freq+50, FFT_noisy[:-1])
            plt.plot(FFT_freq+50, FFT_clean[:-1], 'r')
        plt.yscale('log')
#        plt.savefig("C:\\Users\\User\\Desktop\\FFT.pdf")
        plt.show()
    return ECG_trace_filt_final, noise_freqs, FFT_noisy, FFT_clean

########################################
# Trace math: CWT R-peak finding
########################################

def find_rpeaks_mortara(leads,tr_final_all,optplot):
    """
    Find R-peak locations using CWT on various leads from a directory
    
    Inputs:
        leads-- List of leads (string) 
        tr_final_all-- n by 12 array of traces from each lead in leads at 2000Hz
        optplot-- Boolean flag to display plots
        
    Outputs:
        cwt_peaks-- List of detected R-peak locations
        cwt_leads-- List of leads used to detect R-peaks
        cwt_peak_compare-- List of raw R-peak locations
        cwt_corr_vec-- Vector correlation between R-peak locations
    """
    cwt_peak_unfilt = []
    cwt_peak_all = []
    
    for xx,lead in enumerate([lead for lead in leads if "a" not in lead]):
        # # Find the file index for the lead
        # file_ix = [ix for ix,file in enumerate(files) if lead == file.split('.')[0]][0]
        
        # # Open ECG trace file
        # print('Processing file: %s' % files[file_ix])
        # with open(os.path.join(folder,files[file_ix]), 'r') as tr_file:
        #     ECG_trace = [float(ii) for ii in tr_file.read().split('\n')[:-1]]
            
        print('Processing lead: %s' % lead)
        
        # Isolate the corresponding trace for each lead
        ECG_trace = tr_final_all[:,xx]
            
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))
        
        
        
        # blob = readBLOB(session, cursor, patient, 0, ajbase, lead)
        # if len(blob) == 0 or blob[0] == (None,) or blob[0] == ('',):
        #     print("No "+lead+" found data for patient "+str(patient)+" ("+name+")!")
        #     ECG_trace = []
        #     continue
        # else:
        #     print("Processing lead "+lead+"...")
        # try:
        #     ECG_trace = [float(ii) for ii in blob[0][0].split()]
        # except:
        #     print("Error in "+lead+" data for patient "+str(patient)+" ("+name+")!")
        #     ECG_trace = []
        #     continue
        
        # Skip short traces - n<4 r-peaks can cause issues with PCA etc
        if len(ECG_trace) < 5000:
            print("Insufficient "+lead+" data for patient "+str(patient)+" ("+name+")!")
            continue   
        # Truncate traces longer than 2 minutes to save computing time
        if len(ECG_trace) > 120000:
            print('Length greater than 2 minutes because there are %d data points' % len(ECG_trace))
            # plt.figure()
            # plt.plot(ECG_trace)
            ECG_trace = ECG_trace[:120000]
            # del ECG_trace[120000:]
            # print('Length is now %d data points' % len(ECG_trace))
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))

        # De-noise traces
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace,optplot)
        
        # Normalize traces
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
        
        # CWT Find R-peaks: Bioinformatics (2006) 22 (17): 2059-2065.
        cwt_peak_lead = []
        if "V" in lead:
            cwt_peak_lead = find_QRS_peaks_CWT(ECG_trace,optplot)
        else:
            cwt_peak_lead = find_QRS_peaks_CWT(-ECG_trace,optplot)
        
        # Check for "near-by" peaks that are likely noise
        cwt_peak_del = np.zeros_like(cwt_peak_lead)
        cwt_peak_diff = np.diff(cwt_peak_lead)
        for idx,val in enumerate(cwt_peak_diff):
            if cwt_peak_diff[idx] < 800:
                cwt_peak_del[idx] = 1
        
        # Build lead comparison list
        cwt_peak_unfilt.append(cwt_peak_lead)
        if any(cwt_peak_del):
            cwt_peak_all.append(list(np.delete(cwt_peak_lead, np.argwhere(cwt_peak_del))))
        else:
            cwt_peak_all.append(cwt_peak_lead)
        
    # Find best R-peaks
    if len(ECG_trace) > 5000:
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = rpeak_compare(cwt_peak_all, leads)   
    else:
        cwt_peaks = []
        cwt_leads = []
        cwt_peak_compare = []
        cwt_corr_vec = []
    return cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec

def find_rpeaks_folder(leads,folder,files,optplot):
    """
    Find R-peak locations using CWT on various leads from a directory
    
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        patient-- Patient number
        name-- Patient name (string) - leave as '' if unknown
        ajbase-- String flag (ajm/bas) to select database
        leads-- List of leads (string) 
        optplot-- Boolean flag to display plots
        
    Outputs:
        cwt_peaks-- List of detected R-peak locations
        cwt_leads-- List of leads used to detect R-peaks
        cwt_peak_compare-- List of raw R-peak locations
        cwt_corr_vec-- Vector correlation between R-peak locations
    """
    cwt_peak_unfilt = []
    cwt_peak_all = []
    for xx,lead in enumerate([lead for lead in leads if "a" not in lead]):
        # Find the file index for the lead
        file_ix = [ix for ix,file in enumerate(files) if lead == file.split('.')[0]][0]
        
        # Open ECG trace file
        print('Processing file: %s' % files[file_ix])
        with open(os.path.join(folder,files[file_ix]), 'r') as tr_file:
            ECG_trace = [float(ii) for ii in tr_file.read().split('\n')[:-1]]
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))
        
        
        
        # blob = readBLOB(session, cursor, patient, 0, ajbase, lead)
        # if len(blob) == 0 or blob[0] == (None,) or blob[0] == ('',):
        #     print("No "+lead+" found data for patient "+str(patient)+" ("+name+")!")
        #     ECG_trace = []
        #     continue
        # else:
        #     print("Processing lead "+lead+"...")
        # try:
        #     ECG_trace = [float(ii) for ii in blob[0][0].split()]
        # except:
        #     print("Error in "+lead+" data for patient "+str(patient)+" ("+name+")!")
        #     ECG_trace = []
        #     continue
        
        # Skip short traces - n<4 r-peaks can cause issues with PCA etc
        if len(ECG_trace) < 5000:
            print("Insufficient "+lead+" data for patient "+str(patient)+" ("+name+")!")
            continue   
        # Truncate traces longer than 2 minutes to save computing time
        if len(ECG_trace) > 120000:
            print('Length greater than 2 minutes because there are %d data points' % len(ECG_trace))
            # plt.figure()
            # plt.plot(ECG_trace)
            ECG_trace = ECG_trace[:120000]
            # del ECG_trace[120000:]
            # print('Length is now %d data points' % len(ECG_trace))
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))

        # De-noise traces
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace,optplot)
        
        # Normalize traces
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
        
        # CWT Find R-peaks: Bioinformatics (2006) 22 (17): 2059-2065.
        cwt_peak_lead = []
        if "V" in lead:
            cwt_peak_lead = find_QRS_peaks_CWT(ECG_trace,optplot)
        else:
            cwt_peak_lead = find_QRS_peaks_CWT(-ECG_trace,optplot)
        
        # Check for "near-by" peaks that are likely noise
        cwt_peak_del = np.zeros_like(cwt_peak_lead)
        cwt_peak_diff = np.diff(cwt_peak_lead)
        for idx,val in enumerate(cwt_peak_diff):
            if cwt_peak_diff[idx] < 800:
                cwt_peak_del[idx] = 1
        
        # Build lead comparison list
        cwt_peak_unfilt.append(cwt_peak_lead)
        if any(cwt_peak_del):
            cwt_peak_all.append(list(np.delete(cwt_peak_lead, np.argwhere(cwt_peak_del))))
        else:
            cwt_peak_all.append(cwt_peak_lead)
        
    # Find best R-peaks
    if len(ECG_trace) > 5000:
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = rpeak_compare(cwt_peak_all, leads)   
    else:
        cwt_peaks = []
        cwt_leads = []
        cwt_peak_compare = []
        cwt_corr_vec = []
    return cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec

def find_rpeaks_AW(leads,ECG_trace,optplot):
    """
    Find R-peak locations using CWT on apple watch data
    
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        patient-- Patient number
        name-- Patient name (string) - leave as '' if unknown
        ajbase-- String flag (ajm/bas) to select database
        leads-- List of leads (string) 
        optplot-- Boolean flag to display plots
        
    Outputs:
        cwt_peaks-- List of detected R-peak locations
        cwt_leads-- List of leads used to detect R-peaks
        cwt_peak_compare-- List of raw R-peak locations
        cwt_corr_vec-- Vector correlation between R-peak locations
    """
    cwt_peak_unfilt = []
    cwt_peak_all = []
    for xx,lead in enumerate([lead for lead in leads if "a" not in lead]):
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))
        
        # De-noise traces
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace,optplot)
        
        # Normalize traces
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
        
        # CWT Find R-peaks: Bioinformatics (2006) 22 (17): 2059-2065.
        cwt_peak_lead = []
        if "V" in lead:
            cwt_peak_lead = find_QRS_peaks_CWT(ECG_trace,optplot)
        else:
            cwt_peak_lead = find_QRS_peaks_CWT(-ECG_trace,optplot)
        
        # Check for "near-by" peaks that are likely noise
        cwt_peak_del = np.zeros_like(cwt_peak_lead)
        cwt_peak_diff = np.diff(cwt_peak_lead)
        for idx,val in enumerate(cwt_peak_diff):
            if cwt_peak_diff[idx] < 800:
                cwt_peak_del[idx] = 1
        
        # Build lead comparison list
        cwt_peak_unfilt.append(cwt_peak_lead)
        if any(cwt_peak_del):
            cwt_peak_all.append(list(np.delete(cwt_peak_lead, np.argwhere(cwt_peak_del))))
        else:
            cwt_peak_all.append(cwt_peak_lead)
        
    # Find best R-peaks
    if len(ECG_trace) > 5000:
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = rpeak_compare(cwt_peak_all, leads)   
    else:
        cwt_peaks = []
        cwt_leads = []
        cwt_peak_compare = []
        cwt_corr_vec = []
    return cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec

def find_rpeaks(session, cursor, patient, name, ajbase, leads, optplot):
    """
    Find R-peak locations using CWT on various leads
    
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        patient-- Patient number
        name-- Patient name (string) - leave as '' if unknown
        ajbase-- String flag (ajm/bas) to select database
        leads-- List of leads (string) 
        optplot-- Boolean flag to display plots
        
    Outputs:
        cwt_peaks-- List of detected R-peak locations
        cwt_leads-- List of leads used to detect R-peaks
        cwt_peak_compare-- List of raw R-peak locations
        cwt_corr_vec-- Vector correlation between R-peak locations
    """
    cwt_peak_unfilt = []
    cwt_peak_all = []
    for xx,lead in enumerate([lead for lead in leads if "a" not in lead]):
        # Read in data
        blob = readBLOB(session, cursor, patient, 0, ajbase, lead)
        if len(blob) == 0 or blob[0] == (None,) or blob[0] == ('',):
            print("No "+lead+" found data for patient "+str(patient)+" ("+name+")!")
            ECG_trace = []
            continue
        else:
            print("Processing lead "+lead+"...")
        try:
            ECG_trace = [float(ii) for ii in blob[0][0].split()]
        except:
            print("Error in "+lead+" data for patient "+str(patient)+" ("+name+")!")
            ECG_trace = []
            continue
        
        # Skip short traces - n<4 r-peaks can cause issues with PCA etc
        if len(ECG_trace) < 5000:
            print("Insufficient "+lead+" data for patient "+str(patient)+" ("+name+")!")
            continue   
        # Truncate traces longer than 2 minutes to save computing time
        if len(ECG_trace) > 120000:
            del ECG_trace[120000:]
        ECG_trace = ECG_trace / max(np.abs(ECG_trace))

        # De-noise traces
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace,optplot)
        
        # Normalize traces
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
        
        # CWT Find R-peaks: Bioinformatics (2006) 22 (17): 2059-2065.
        cwt_peak_lead = []
        if "V" in lead:
            cwt_peak_lead = find_QRS_peaks_CWT(ECG_trace,optplot)
        else:
            cwt_peak_lead = find_QRS_peaks_CWT(-ECG_trace,optplot)
        
        # Check for "near-by" peaks that are likely noise
        cwt_peak_del = np.zeros_like(cwt_peak_lead)
        cwt_peak_diff = np.diff(cwt_peak_lead)
        for idx,val in enumerate(cwt_peak_diff):
            if cwt_peak_diff[idx] < 800:
                cwt_peak_del[idx] = 1
        
        # Build lead comparison list
        cwt_peak_unfilt.append(cwt_peak_lead)
        if any(cwt_peak_del):
            cwt_peak_all.append(list(np.delete(cwt_peak_lead, np.argwhere(cwt_peak_del))))
        else:
            cwt_peak_all.append(cwt_peak_lead)
        
    # Find best R-peaks
    if len(ECG_trace) > 5000:
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = rpeak_compare(cwt_peak_all, leads)   
    else:
        cwt_peaks = []
        cwt_leads = []
        cwt_peak_compare = []
        cwt_corr_vec = []
    return cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec

def find_QRS_peaks_CWT(ECG_trace, optplot):
    """
    CWT Find peaks: Bioinformatics (2006) 22 (17): 2059-2065.
    find_peaks_cwt(vector, widths, wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=1, noise_perc=10)
    
    Inputs:
        ECG_trace-- Input data (array)
        optplot-- Boolean flag to display plots
        
    Output:
        cwt_peaks-- List of detected R-peak locations
    """
    cwt_peaks = sig.find_peaks_cwt(-ECG_trace, np.arange(4,5), gap_thresh=None, min_snr=3) 
    # Filter CWT peaks for negative and close to beginning/end
    cwt_peaks = [peak for peak in cwt_peaks if \
                 ECG_trace[peak] <= 0 and peak > 1000 and peak < len(ECG_trace)-1000]
    # Plot
    if optplot:
        plt.figure(figsize=(12,6))
        plt.plot(ECG_trace)
        plt.xlim(0,10000)
    # Basin hopping minimization routine
    iter0 = 2
    window = 50
    for _ in range(iter0):
        for cwt_ix,cwt_peak in enumerate(cwt_peaks):
            ECG_window = ECG_trace[cwt_peak-window:cwt_peak+window]
            offset = window - np.argmin(ECG_window)
            cwt_peaks[cwt_ix] = cwt_peaks[cwt_ix] - offset
    # Remove duplicates
    cwt_peaks = list(set(cwt_peaks))
    cwt_peaks.sort()
    iter1 = 3
    for _ in range(iter1):
        # Combine peaks to the right less than heart rate (false p-wave detect)    
        for cwt_ix in range(len(cwt_peaks)-1):
            if cwt_peaks[cwt_ix+1]-cwt_peaks[cwt_ix] < 600:
                cwt_peaks[cwt_ix] = cwt_peaks[cwt_ix+1]
        cwt_peaks = list(set(cwt_peaks))
        cwt_peaks.sort()
    # Plot
    if optplot:
        plt.scatter(cwt_peaks, ECG_trace[cwt_peaks], s=80, facecolors='none', edgecolors='r')
        plt.show()
    return cwt_peaks
    
def beat_chop(ECG_trace, cwt_peaks):
    """
    Chop up trace on R-peaks
    
    Inputs:
        ECG_trace-- Linear ECG trace data (array)
        cwt_peaks-- List of detected R-peak locations
        
    Output:
        ECG_stack-- Stack of chopped and re-centered ECG traces
    
    """
    ECG_stack = np.zeros(shape=(len(cwt_peaks),1500))
    for stack_ix,rpeak in enumerate(cwt_peaks):
        ECG_stack[stack_ix,:] = ECG_trace[rpeak-600:rpeak+900] - np.median(ECG_trace[rpeak-600:rpeak+900])
    return ECG_stack

def rpeak_compare(cwt_peak_all, leads):
    """
    Generate list of R-peak locations based on traces
    
    Inputs:
        cwt_peak_all-- List of all detected R-peak locations
        leads-- List of leads (string)
        
    Output:
        cwt_peaks-- List of refined detected R-peak locations
        cwt_leads-- List of leads used to detect R-peaks
        cwt_peak_compare-- List of raw R-peak locations
        cwt_corr_vec-- Vector correlation between R-peak locations
    """
    cwt_leads = []
    # Find good leads based on similar R-peaks
    cwt_leads = [xx for xx,peaks in enumerate(cwt_peak_all) if \
        len(peaks) == stats.mode([len(xx) for xx in cwt_peak_all if len(xx)>1])[0][0]]
    cwt_peak_compare = [cwt_peak_all[xx] for xx in cwt_leads]
    cwt_leads = [leads[xx] for xx in cwt_leads]
    cwt_peak_compare = np.array(cwt_peak_compare).astype(int)
    
    # Calculate vector correlation between R-peak positions
    try:
        cwt_corr_vec = sum(np.corrcoef(cwt_peak_compare)) / cwt_peak_compare.shape[0]
    except:
        print("No common mode among R-peak counts.")
        cwt_corr_vec = np.zeros([1])
    
    # Check for offset in some leads
    cwt_offset_check = abs(np.median(cwt_peak_compare[:,0])-cwt_peak_compare[:,0])
    if any(cwt_offset_check>1000):
        if any(cwt_offset_check<1000): # If there are any non-offset leads
            cwt_peak_compare = np.delete(cwt_peak_compare,np.where(cwt_offset_check>1000),axis=0)
            try:
                cwt_corr_vec = sum(np.corrcoef(cwt_peak_compare))/cwt_peak_compare.shape[0]
            except:
                print("No common mode among R-peak counts.")
                cwt_corr_vec = np.zeros([1])
        else: # If all leads are offset, take the first
            cwt_peak_compare = np.delete(cwt_peak_compare, \
                                         np.where(cwt_peak_compare[:,0] != np.amin(cwt_peak_compare[:,0])) \
                                         ,axis=0)
            try:
                cwt_corr_vec = sum(np.corrcoef(cwt_peak_compare))/cwt_peak_compare.shape[0]
            except:
                print("No common mode among R-peak counts.")
                cwt_corr_vec = np.zeros([1])
    
        # Use maximum number of peak counts if correlation is not good
    if len(cwt_peak_compare) > 1:
        if not any(cwt_corr_vec > 0.99): 
            if any(cwt_corr_vec != 0): # Cases with a second mode
                print("Poor corrleation between R-peak locations in first mode of counts.")
            cwt_leads = [xx for xx,peaks in enumerate(cwt_peak_all) if \
                len(peaks) == max([len(xx) for xx in cwt_peak_all if len(xx)>1 and \
                    len(xx) != stats.mode([len(xx) for xx in cwt_peak_compare if len(xx)>1])])]
            cwt_peak_compare = [cwt_peak_all[xx] for xx in cwt_leads]
            cwt_leads = [leads[xx] for xx in cwt_leads]
            # Offset fix
            if any(cwt_offset_check>1000):
                cwt_leads_filt = []
                for ii in np.argwhere(cwt_offset_check>1000).tolist():
                    cwt_leads_filt.append(cwt_leads[ii[0]])
                cwt_leads = cwt_leads_filt
            cwt_peak_compare = np.array(cwt_peak_compare).astype(int)
            try: # Second mode
                if not any(cwt_corr_vec > 0.99): # Take first list from 2nd mode if correlation is not good
                    if any(cwt_corr_vec == 0): # Cases with no common mode
                        raise
                    print("Using first list from second mode of counts.")
                    cwt_peaks = list(cwt_peak_compare[0])
                else: # Take correlated mean of 2nd mode
                    print("Using second mode of R-peak counts.")
                    cwt_peaks = list(np.floor(np.mean(cwt_peak_compare[[xx for xx,corr in \
                        enumerate(cwt_corr_vec) if corr > 0.99],:],axis=0)).astype(int))
            except: # No second mode
                print("Using maximum value of R-peak counts.")
                if len(cwt_peak_compare) == 1: # Take first and only from maximum
                    cwt_peaks = list(cwt_peak_compare[0])
                else: # Take mean of maximum (probably unnecessary)
                    cwt_peaks = list(np.floor(np.mean(cwt_peak_compare,axis=0)).astype(int))
        else: # Take correlated median <<<<<<<<<<<<<<<<<<<<<
            cwt_peaks = np.median(cwt_peak_compare[[xx for xx,corr in \
                        enumerate(cwt_corr_vec) if corr > 0.99],:],axis=0).astype(int)
    else:
        print("Using only one lead to locate R-peaks.")
        cwt_peaks = list(cwt_peak_compare[0])
        cwt_leads = leads[list(np.where(cwt_peak_compare[:,0] == np.amin(cwt_peak_compare[:,0]))[0])[0]]
        
#    # Remove last R-peak to prevent short clipping
#    if len(cwt_peaks) > 4:
#        del cwt_peaks[-1]
    return cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec

########################################
# Trace math: PCA outlier exclusion
########################################

def remove_outlier_traces(ECG_stack, max_dist, min_samples, optplot=False):
    """
    Identify outlier trace segments using PCA
    
    Inputs:
        ECG_stack-- Stacked ECG trace data (array)
        max_dist-- Maximum distance between PCA clusters (sklearn.cluster.DBSCAN)
        min_samples--Minimum number of traces per PCA cluster (sklearn.cluster.DBSCAN)
        optplot-- Boolean flag to display plots
        
    Outputs:
        ECG_stack-- Unchanged from input
        ECG_stack_fix-- Outlier-excluded stacked ECG trace data
        pct_outier-- Percent of traces in stack counted as outliers
        tr_outliers-- Index of traces in stack counted as outliers
        clusters-- Numeric cluster labels for traces in stack
    """
    from sklearn.decomposition import PCA
    from scipy.stats import f
    from sklearn.cluster import DBSCAN
    # Normalize
    X = np.zeros_like(ECG_stack)
    ii = 0
    for row in ECG_stack:
        row = row - min(row) / (max(row)-min(row))
        X[ii,:] = row
        ii += 1
    # Perform PCA
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X)
    # Calculate Hotelling confidence zones (at 95%) for each component
    confidence = []
    for comp in range(3):
        comp += 1
        confidence.append(np.sqrt(np.var(scores[:,comp-1]) * \
                 (comp*(X.shape[1]**2-1)/(X.shape[1]*(X.shape[1]-comp)))*f.isf(0.05,comp,X.shape[1]-comp)))
    conf_x = np.cos(np.linspace(0,2*np.pi,num=100))*confidence[0]
    conf_y = np.sin(np.linspace(0,2*np.pi,num=100))*confidence[1]
    # Find segments falling outside confidence zone
    conf_outliers = [ii for ii,xx in enumerate(in_hull(scores[:,:2], np.array([conf_x, conf_y]).T)) if not xx]
    # Calculate object correlation coefficients #### NEW
    corr_mtx = np.corrcoef(X)
    corr_vec = sum(corr_mtx)/X.shape[0]
    # Calculate object leverage
    leverage = []
    for obj in range(X.shape[0]):
        leverage.append(1/X.shape[0] + sum([scores[obj,comp]**2 / np.dot(scores[:,comp].T, \
                                           scores[:,comp]) for comp in range(3)]))
    conf_outliers += [ii for ii,xx in enumerate(leverage) if xx>np.mean(leverage)+2*np.std(leverage)]
    # Analyze clusters in PCA score plot
    db = DBSCAN(eps=max_dist, min_samples=min_samples).fit(scores[:,:2])
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    conf_outliers_spot = conf_outliers
    n_outliers = len(set(conf_outliers))
    # Count clusters and select largest - CHANGED 
    if n_clusters >= 1:
#    if n_clusters > 1:
        counts = []
        for cluster in range(n_clusters):
            counts.append(list(labels).count(cluster))
        conf_outliers_spot += [ii for ii,xx in enumerate(labels) if xx != np.argmax(counts)]
    conf_outliers_spot.sort(key=int)
    conf_outliers_spot = set(conf_outliers_spot)
    if optplot:
        stats_plot(ECG_stack, conf_outliers, conf_outliers_spot, scores, core_samples_mask, \
                   conf_x, conf_y, leverage, corr_vec, labels, n_clusters)
    return ECG_stack, np.array([xx for ii,xx in enumerate(ECG_stack) if ii not in conf_outliers_spot]), \
             n_outliers/ECG_stack.shape[0], list(conf_outliers_spot), labels

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def stats_plot(ECG_stack, conf_outliers, conf_outliers_spot, scores, core_samples_mask, \
               conf_x, conf_y, leverage, corr_vec, labels, n_clusters):
    """
    Plot ECG trace outlier statistics using PCA
    
    Inputs:
        ECG_stack-- Stacked ECG trace data (array)
        conf_outliers-- List of outlier ECG traces in stack using PCA
        conf_outliers_spot--  List of outlier ECG traces in stack using PCA and cluster analysis
        scores-- PCA scores for trace stack
        core_samples_mask-- Core traces from cluster analysis
        conf_x-- Hotelling 95% confidence region (PC1)
        conf_y-- Hotelling 95% confidence region (PC2)
        leverage-- Object leverage for each trace
        corr_vec-- Vector correlation between traces
        labels-- Numeric cluster labels
        n_clusters_-- Number of clusters detected
        
    Outputs:
        none
    """
    # PCA plot
    fig = plt.figure(figsize = (12,5))
    ax = fig.add_subplot(1,2,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA Score Plot', fontsize = 20)
    ax.plot(conf_x, conf_y)
    plt.fill_between(conf_x, conf_y, 0, color='green', alpha='0.15')
    ax.scatter(scores[:,0], scores[:,1], s = 50)
    ax.scatter(scores[conf_outliers,0], scores[conf_outliers,1], s = 50, color='red')
    ax.grid()        
    # Leverage/correlation plot
    ax2 = fig.add_subplot(1,2,2) 
    ax2.set_xlabel('Object Leverage', fontsize = 15)
    ax2.set_ylabel('Vector Correlation', fontsize = 15)
    ax2.set_title('Leverage/Corr Plot', fontsize = 20)
    plt.fill_between(np.linspace(0,np.mean(leverage)+2*np.std(leverage)), \
                1, np.mean(corr_vec)-np.std(corr_vec), color='green', alpha='0.15')
    ax2.scatter(leverage, corr_vec, s = 50)
    ax2.scatter([leverage[ii] for ii in conf_outliers], \
                [corr_vec[ii] for ii in conf_outliers], s = 50, color='red')
    ax2.grid()
    # Cluster plot
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
#    fig.savefig("C:\\Users\\User\\Desktop\\stats1.pdf")
    
    fig = plt.figure(figsize=(5,5))
    for k, col in zip(unique_labels, colors):
        if k == -1: # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = scores[:,:2][class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        xy = scores[:,:2][class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    plt.plot(conf_x, conf_y)
    plt.fill_between(conf_x, conf_y, 0, color='green', alpha='0.15')
    plt.grid()    
    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()
#    fig.savefig("C:\\Users\\User\\Desktop\\stats2.pdf")
    # Trace comparison
    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot(2,1,1)
    _=ax.plot(ECG_stack.T)
    ax2 = fig.add_subplot(2,1,2)
    _=ax2.plot(np.array([xx for ii,xx in enumerate(ECG_stack) if ii not in conf_outliers_spot]).T)
#    plt.savefig("C:\\Users\\User\\Desktop\\tracestack.pdf")
    plt.show()

########################################
# Trace math: Resampling
########################################
    
def resample_2kHz(ECG_trace, tr_freq, wavelet):
    """
    Resamples an ECG to 2000 Hz using DWT
    
    Inputs:
        ECG_trace-- Linear ECG trace data (array)
        tr_freq-- Input trace frequency (Hz)
        wavelet-- Wavelet (string) to use for DWT
        
    Output:
        ECG_trace_resampled-- ECG trace data resampled to 2000 Hz
    """
    import pywt
    # Time elapsed
    tr_time = len(ECG_trace)/tr_freq
    # Decompose ECG to wavelet basis
    (cA, cD) = pywt.dwt(ECG_trace, wavelet)
    # Remove zero padding for odd number sampling frequencies
    if (len(ECG_trace) % 2) == 1:
        cD[-1] = cD[-2]
    # Dilate wavelet coefficients to 2000 Hz
    cA = sig.resample(cA, int(1000*tr_time))
    cD = sig.resample(cD, int(1000*tr_time)) * 2000/tr_freq
    # Inverse DWT back to original time basis
    trace_2000 = pywt.idwt(cA, cD, wavelet, 'smooth')
    mm_to_um = 1 #1000
    return trace_2000*mm_to_um

def resample_varrate(ECG_trace, tr_freq, out_freq):
    """
    Resamples an ECG to the desired frequency
        
    Inputs:
        ECG_trace-- Linear ECG trace data (array)
        tr_freq-- Input frequency (Hz)
        out_freq-- Desired output frequency (Hz)
        
    Output:
        ECG_trace_resampled-- ECG trace data resampled to out_freq
    """
    out_len = len(ECG_trace) * out_freq/tr_freq
    tr_filt = sig.resample(ECG_trace, out_len, t=None, axis=1, window=None)
    return tr_filt

def generate_median_fusion(tr_final_all,cwt_peaks,leads):
    print('Chopping leads...')
    # calculate the average heart beat for each lead using CWT peak finder
    ECG_medians,noise_level_raw,noise_level_FFT,pct_outliers = list(),list(),list(),list()                       
    for lead_ix,lead in enumerate(leads):
        # print('Processing lead: %s' % lead)
        # Normalize trace
        ECG_trace = tr_final_all[:,lead_ix] / max(np.abs(tr_final_all[:,lead_ix]))
    
        # Denoise trace
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace, optplot=False)
        
        # Calculate noise levels (coeff of variation) from FFT
        noise_level_raw.append(np.std(FFT_noisy[int(len(FFT_noisy)*.75):]) / \
            np.mean(FFT_noisy[int(len(FFT_noisy)*.75):]))
        noise_level_FFT.append(np.std(FFT_clean[int(len(FFT_clean)*.75):]) / \
            np.mean(FFT_clean[int(len(FFT_clean)*.75):]))
             
        # Normalize trace            
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
    
        # Chop and stack on R-peak
        if len(cwt_peaks) != 0:
            ECG_stack = beat_chop(ECG_trace, cwt_peaks)
            # Apply PCA, other trace math
            [_, ECG_stack_fix, pct_outlier, _, labels] = \
                    remove_outlier_traces(ECG_stack, max_dist=5, min_samples=10, optplot=False) #.5,10
            pct_outliers.append(pct_outlier)
            ECG_median = np.median(ECG_stack_fix, axis=0)  
            ECG_medians.append(ECG_median)
            
    #% final fusion trace
    tr_final = np.reshape(np.reshape(np.asarray(ECG_medians),(12*1500,)),(12*1500,)) 
    trace = np.copy(ECG_norm_resample(tr_final))
    print('Chopping Complete')

    return tr_final,trace,noise_level_raw,noise_level_FFT,pct_outliers

def generate_median_fusion_AW(tr_final_all,cwt_peaks,leads):
    """
    generate median fusion for apple watch
    """
    print('Chopping leads...')
    # calculate the average heart beat for each lead using CWT peak finder
    ECG_medians,noise_level_raw,noise_level_FFT,pct_outliers = list(),list(),list(),list()                       
    for lead_ix,lead in enumerate(leads):
        # print('Processing lead: %s' % lead)
        # Normalize trace
        ECG_trace = tr_final_all / max(np.abs(tr_final_all))
    
        # Denoise trace
        [noise_freqs, FFT_noisy, FFT_freq] = ECG_FFT(ECG_trace, 25)
        [ECG_trace, noise_freqs, FFT_noisy, FFT_clean] = ECG_filt(ECG_trace, optplot=False)
        
        # Calculate noise levels (coeff of variation) from FFT
        noise_level_raw.append(np.std(FFT_noisy[int(len(FFT_noisy)*.75):]) / \
            np.mean(FFT_noisy[int(len(FFT_noisy)*.75):]))
        noise_level_FFT.append(np.std(FFT_clean[int(len(FFT_clean)*.75):]) / \
            np.mean(FFT_clean[int(len(FFT_clean)*.75):]))
             
        # Normalize trace            
        ECG_trace = (ECG_trace-ECG_trace.mean()) / ECG_trace.std()
    
        # Chop and stack on R-peak
        if len(cwt_peaks) != 0:
            ECG_stack = beat_chop(ECG_trace, cwt_peaks)
            # Apply PCA, other trace math
            [_, ECG_stack_fix, pct_outlier, _, labels] = \
                    remove_outlier_traces(ECG_stack, max_dist=5, min_samples=10, optplot=False) #.5,10
            pct_outliers.append(pct_outlier)
            ECG_median = np.median(ECG_stack_fix, axis=0)  
            ECG_medians.append(ECG_median)
            
    #% final fusion trace
    tr_final = np.reshape(np.reshape(np.asarray(ECG_medians),(1500,)),(1500,)) 
    trace = np.copy(ECG_norm_resample(tr_final))
    print('Chopping Complete')

    return tr_final,trace,noise_level_raw,noise_level_FFT,pct_outliers

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

#%%

########################################
# SQL Queries
########################################

def sql_reset(session, cursor):
    """
    Erase data and reinitalize SQL database
        
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        
    Outputs:
        none    
    """
    
    sql_file = """UNLOCK TABLES;
    USE brugada;
    
    DROP TABLE IF EXISTS `brugada`.`patient_info`;
    CREATE TABLE `brugada`.`patient_info` (
        `PatientID` INT(11) NOT NULL AUTO_INCREMENT,
        `PatientName` VARCHAR(40) DEFAULT NULL,
        `Diagnosis` INT DEFAULT NULL,
    	`Sex` CHAR(1) DEFAULT NULL,
        `Age` FLOAT UNSIGNED DEFAULT NULL,
        `Syncope` INT(1) DEFAULT NULL,
        `Cardiac_Arrest` INT(1) DEFAULT NULL,
        `FamHist_SCD` INT(1) DEFAULT NULL,
        `FamHist_BrS` INT(1) DEFAULT NULL,
        `ECGType` VARCHAR(20) DEFAULT NULL,
        `V1_Outlier_Pct` FLOAT UNSIGNED DEFAULT NULL,
        `Baseline_HR` FLOAT UNSIGNED DEFAULT NULL,
        `Ajmaline_HR` FLOAT UNSIGNED DEFAULT NULL,
    	PRIMARY KEY (`PatientID`)
    )  ENGINE=MYISAM AUTO_INCREMENT=1 DEFAULT CHARSET=LATIN1 COMMENT='Basic patient information';
    
    DROP TABLE IF EXISTS `brugada`.`baseline_traces`;
    CREATE TABLE `brugada`.`baseline_traces` (
    	`PatientID` INT(11) UNSIGNED NOT NULL DEFAULT '0',
        `Diagnosis` INT DEFAULT NULL,
        `Session` INT(5) DEFAULT NULL,
        `Page` INT(5) DEFAULT NULL,
        `I` LONGBLOB DEFAULT NULL,
        `II` LONGBLOB DEFAULT NULL,
        `III` LONGBLOB DEFAULT NULL,
        `V1` LONGBLOB DEFAULT NULL,
        `V2` LONGBLOB DEFAULT NULL,
    	`V3` LONGBLOB DEFAULT NULL,
        `V4` LONGBLOB DEFAULT NULL,
        `V5` LONGBLOB DEFAULT NULL,
        `V6` LONGBLOB DEFAULT NULL,
    	`aVF` LONGBLOB DEFAULT NULL,
        `aVL` LONGBLOB DEFAULT NULL,
        `aVR` LONGBLOB DEFAULT NULL,
    	PRIMARY KEY (`PatientID`)
    )  ENGINE=MYISAM ;
    
    DROP TABLE IF EXISTS `brugada`.`baseline_traces_chopped`;
    CREATE TABLE `brugada`.`baseline_traces_chopped` LIKE `brugada`.`baseline_traces`;
    
    ALTER TABLE  `brugada`.`baseline_traces_chopped`
    ADD `QRS_peaks` BLOB DEFAULT NULL,
    ADD `QRS_leads` VARCHAR(30) DEFAULT NULL;
    
    DROP TABLE IF EXISTS `brugada`.`baseline_traces_prePCA`;
    CREATE TABLE `brugada`.`baseline_traces_prePCA` LIKE `brugada`.`baseline_traces`;
    
    ALTER TABLE  `brugada`.`baseline_traces_prePCA`
    ADD `avg_noise_level_raw` FLOAT UNSIGNED DEFAULT NULL,
    ADD `avg_noise_level_FFT` FLOAT UNSIGNED DEFAULT NULL,
    ADD `avg_outlier_pct` FLOAT UNSIGNED DEFAULT NULL,
    ADD `n_pca_clusters` INT(5) DEFAULT NULL;
    
    DROP TABLE IF EXISTS `brugada`.`baseline_traces_representative`;
    CREATE TABLE `brugada`.`baseline_traces_representative` LIKE `brugada`.`baseline_traces`;
    
    DROP TABLE IF EXISTS `brugada`.`ajmaline_traces`;
    CREATE TABLE `brugada`.`ajmaline_traces` LIKE `brugada`.`baseline_traces`;
    
    DROP TABLE IF EXISTS `brugada`.`ajmaline_traces_chopped`;
    CREATE TABLE `brugada`.`ajmaline_traces_chopped` LIKE `brugada`.`baseline_traces_chopped`;
    
    DROP TABLE IF EXISTS `brugada`.`ajmaline_traces_prePCA`;
    CREATE TABLE `brugada`.`ajmaline_traces_prePCA` LIKE `brugada`.`baseline_traces_prePCA`;
    
    DROP TABLE IF EXISTS `brugada`.`ajmaline_traces_representative`;
    CREATE TABLE `brugada`.`ajmaline_traces_representative` LIKE `brugada`.`baseline_traces`;"""
    
    sql_commands = sql_file.split(";")
    for jj, command in enumerate(sql_commands):
        if command != "":    
            try:
                cursor.execute(command, )
            except:
                print("Command skipped at line",jj,":\n",command)

def sql_copy(session, cursor):
    """
    Copy patient info between SQL tables
        
    Inputs:
        session-- SQL session (see mysql.connector.connect for syntax)
        cursor-- SQL cursor
        
    Outputs:
        none
    """
    
    sql_cuttr_query = """INSERT INTO baseline_traces_chopped (PatientID,Diagnosis,Session,Page)
                    SELECT PatientID,Diagnosis,Session,Page FROM baseline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Baseline ECG info not copied: {}".format(error))
    sql_cuttr_query = """INSERT INTO baseline_traces_prePCA (PatientID,Diagnosis,Session,Page)
                        SELECT PatientID,Diagnosis,Session,Page FROM baseline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Baseline ECG info not copied: {}".format(error))
    sql_cuttr_query = """INSERT INTO baseline_traces_representative (PatientID,Diagnosis,Session,Page)
                        SELECT PatientID,Diagnosis,Session,Page FROM baseline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Baseline ECG info not copied: {}".format(error))
    
    sql_cuttr_query = """INSERT INTO ajmaline_traces_chopped (PatientID,Diagnosis,Session,Page)
                        SELECT PatientID,Diagnosis,Session,Page FROM ajmaline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Ajmaline ECG info not copied: {}".format(error))
    sql_cuttr_query = """INSERT INTO ajmaline_traces_prePCA (PatientID,Diagnosis,Session,Page)
                        SELECT PatientID,Diagnosis,Session,Page FROM ajmaline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Ajmaline ECG info not copied: {}".format(error))
    sql_cuttr_query = """INSERT INTO ajmaline_traces_representative (PatientID,Diagnosis,Session,Page)
                        SELECT PatientID,Diagnosis,Session,Page FROM ajmaline_traces;"""
    try:
        cursor.execute(sql_cuttr_query,)
        session.commit()
    except mysql.connector.Error as error:
        session.rollback()
        print("Ajmaline ECG info not copied: {}".format(error))
  
    
#%%
        
########################################
# Trace math: MODWT R-peak finding (depreciated)
########################################

#def find_QRS_peaks_MODWT(trace):
#    """
#    Locate R-peaks in trace using MO-DWT
#    """
#    trace = trace/max(abs(trace))
#    wt = modwt(trace, 'db9', 8)
#    discard = [0,1,2,3,4,5,8]
#    for jj in list(discard):
#        wt[jj] = np.zeros_like(wt[jj])
#    dwt = imodwt(wt, 'db9')
#    dwt = abs(dwt)**2
#    peak_thresh = np.mean(dwt) + 4*np.std(dwt)
#    # Approximate R peaks location
#    [QRS_approx,_] = sig.find_peaks(dwt, height=peak_thresh, distance=250)
#    # Re-center detected R peaks
#    QRS_locs = []
#    for peak in QRS_approx[1:-1]:
#        tr_seg = trace[peak-250:peak+250]
#        tr_seg = tr_seg - tr_seg[0]
#        [QRS_temp,QRS_idx] = sig.find_peaks(abs(tr_seg)**2, height=peak_thresh, distance=50)
#        if not QRS_temp.any():
#            QRS_temp = max(abs(tr_seg**2))
#        QRS_locs.append(QRS_temp[np.argmax([QRS_idx['peak_heights'] for ii in QRS_idx])] + peak-250 + 1)
#    # Uniformize sign of R peak detection
#    QRS_peaks = [ii for ii in trace[QRS_locs]]
#    iteration = 0
#    while np.ptp(np.sign(QRS_locs)) > 0 and iteration < 10:
#        [QRS_locs_uniform] = uniform_rpeaks(trace, peak_thresh, QRS_peaks, QRS_locs, iteration)
#        QRS_locs = QRS_locs_uniform
#        QRS_peaks = [ii for ii in trace[QRS_locs]]
#        iteration += 1
#    if iteration == 10:
#        print("Maximum iterations of R peak sign uniformization process exceeded for baseline V1 lead. Verify quality of ECG trace.")
#    return QRS_locs, QRS_peaks, stat.mode(np.sign(QRS_peaks))
#
#def uniform_rpeaks(trace, thresh, QRS_peaks, QRS_locs, iteration): # Check iter here
#    """
#    Uniformize sign of R peak detection
#    """
#    QRS_locs_uniform = []
#    for ii,loc in enumerate(QRS_locs):
#        if stat.mode(np.sign(QRS_peaks)) == 1:
#            if np.sign(QRS_peaks[ii]) == -1:
#                [QRS_temp,QRS_idx] = sig.find_peaks(abs(trace[loc-250:loc+250])**2, height=thresh, distance=50)
#                if iteration == 0 and len(QRS_temp) > 1:
#                    del QRS_temp[QRS_idx]
#                    QRS_idx = max(QRS_temp)
#                QRS_locs_uniform.append(QRS_temp[np.argmax([QRS_idx['peak_heights'] for ii in QRS_idx])] + loc-250)
#            else:
#                QRS_locs_uniform.append(loc)
#        elif stat.mode(np.sign(QRS_peaks)) == -1:
#            if np.sign(QRS_peaks[ii]) == 1:
#                [QRS_temp,QRS_idx] = sig.find_peaks(abs(trace[loc-250:loc+250])**2, height=thresh, distance=50)
#                if iteration == 0 and len(QRS_temp) > 1:
#                    del QRS_temp[QRS_idx]
#                    QRS_idx = max(QRS_temp)
#                QRS_locs_uniform.append(QRS_temp[np.argmax([QRS_idx['peak_heights'] for ii in QRS_idx])] + loc-250)
#            else:
#                QRS_locs_uniform.append(loc)
#    return QRS_locs_uniform
#
#    
#def segment_traces(trace, locs, width):
#    """
#    Segment traces based on R peak locations
#    """
#    midpoints = np.floor(locs[:-1] + np.diff(locs)/2)
#    R_width = np.diff(midpoints)
#    cut_tr = np.array([])
#    # Build array of segmented traces
#    for loc in locs[1:-1]:
#        tr_out = trace[int(np.ceil(loc-2*width/5)):int(np.floor(loc+3*width/5))]
#        if not cut_tr.any():
#            cut_tr = tr_out
#        else:
#            cut_tr = np.vstack([cut_tr, tr_out])
#    # Center segments on R peak (at x=600)
#    offset = []
#    cut_tr_shift = np.array([])
#    for tr_seg in cut_tr:
#        try:
#            [QRS_temp,_] = sig.find_peaks(abs(tr_seg)**2, height=np.mean(tr_seg) + 2*np.std(tr_seg), distance=500)
#            offset.append(int(QRS_temp))
#        except:
#            offset.append(max(abs(tr_seg)**2))
#    cut_tr_shift = np.roll(cut_tr, int(np.floor(2*width/5-np.median(offset))))
#    # Reject extra-short or extra-long segments
#    keep = []
#    for ii,binval in enumerate((abs(R_width-np.mean(R_width)) > np.std(R_width)*2).astype(int)):
#        if not binval:
#            keep.append(ii)
#    cut_tr_out = [cut_tr_shift[ii] for ii in keep]
#    cut_tr_out = np.array(cut_tr_out).astype(np.float)
#    return cut_tr_out