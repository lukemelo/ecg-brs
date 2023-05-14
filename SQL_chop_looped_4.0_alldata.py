"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada SQL trace-chopping script
%
% Ashton Christy
# Luke Melo
% 3 Jun 2019
% 
% Version 4.0
% Edited 17 Dec 2020
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Load packages
import numpy as np, mysql.connector, time, ecglib as ecg

t0 = time.time()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
            
# Initialize MySQL session
session = mysql.connector.connect(host='localhost',
                             database='brugada',
                             user='root',
                             password='root',
                             use_pure=True)
cursor = session.cursor(buffered=True)
sql_fetch_patients = """SELECT PatientID FROM `patient_info`;"""
cursor.execute(sql_fetch_patients, )
patient_name = ''
patient_id = [ii for ii in cursor.fetchall()]
patient_id = [ii[0] for ii in patient_id]
leads = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]
datasets = ["baseline", "ajmaline"]

for idx,patient in enumerate(patient_id):
        
    print("\n-----------------------------------------------")
    print("Processing patient "+str(patient)+" - "+str(idx+1)+" of "+str(max(patient_id))+"...")
        
    for ajbase in datasets:
        print("\nProcessing "+ajbase+" data...")
        cursor.execute("SELECT COUNT(*) FROM `"+ajbase+"_traces` WHERE `PatientID` = """+str(patient)+""";""", )
        if not cursor.fetchall()[0][0]:
            session.rollback()
            print("No "+ajbase+" data found for patient "+str(patient)+"!  Skipping...") 
            continue
        
        # Get data
        [cwt_peaks, cwt_leads, cwt_peak_compare, cwt_corr_vec] = \
            ecg.find_rpeaks(session, cursor, patient, patient_name, ajbase, leads, optplot=False)
        
        # Calculate heart rate
        heart_rate = 120000/np.median(np.diff(cwt_peaks))
        if heart_rate >= 120:
            print("Excessive heart rate ("+'{0:.1f}'.format(heart_rate)+") detected for patient "+str(patient)+"!\nData not processed.")
            continue
    
        pct_outliers = []
        noise_level_raw = []
        noise_level_FFT = []
        
        for lead in leads:
            try:
                blob = ecg.readBLOB(session, cursor, patient, 0, ajbase, lead)
                ECG_trace = [float(ii) for ii in blob[0][0].split()]
                ECG_trace = ECG_trace / max(np.abs(ECG_trace))
                
                # Denoise trace
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
                
                # Update SQL
                if "bas" in ajbase.lower():
                    sql_QRS_update = """ UPDATE `baseline_traces_chopped`
                                        SET `"""+lead+"""` = (%s), `QRS_peaks` = (%s), `QRS_leads` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""  
                    sql_PCA_update = """ UPDATE `baseline_traces_prePCA`
                                        SET `"""+lead+"""` = (%s), `avg_noise_level_raw` = (%s), `avg_noise_level_FFT` = (%s), `avg_outlier_pct` = (%s), `n_pca_clusters` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""                     
                    sql_rep_update = """ UPDATE `baseline_traces_representative`
                                        SET `"""+lead+"""` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""   
                    sql_hr_update = """ UPDATE `patient_info`
                                        SET `Baseline_HR` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""                                             
                elif "ajm" in ajbase.lower():
                    sql_QRS_update = """ UPDATE `ajmaline_traces_chopped`
                                        SET `"""+lead+"""` = (%s), `QRS_peaks` = (%s), `QRS_leads` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""
                    sql_PCA_update = """ UPDATE `ajmaline_traces_prePCA`
                                        SET `"""+lead+"""` = (%s), `avg_noise_level_raw` = (%s), `avg_noise_level_FFT` = (%s), `avg_outlier_pct` = (%s), `n_pca_clusters` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""                      
                    sql_rep_update = """ UPDATE `ajmaline_traces_representative`
                                        SET `"""+lead+"""` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""   
                    sql_hr_update = """ UPDATE `patient_info`
                                        SET `Ajmaline_HR` = (%s)
                                        WHERE `PatientID` = """+str(patient)+""";"""        

                cursor.execute(sql_QRS_update, (ECG_stack_fix.tostring(), \
                                                np.array(cwt_peaks).tostring(), \
                                                str(' '.join(cwt_leads)), ))
                cursor.execute(sql_PCA_update, (ECG_stack.tostring(), \
                                                float(np.mean(noise_level_raw)), \
                                                float(np.mean(noise_level_FFT)), \
                                                float(np.mean(pct_outliers)), \
                                                (len(set(labels)) - (1 if -1 in labels else 0)), ))
                cursor.execute(sql_rep_update, (ECG_median.tostring(), ))
                cursor.execute(sql_hr_update, (float(heart_rate), ))
                session.commit()
                print("Lead "+lead+" processed.")
                
            except:
                session.rollback()
                print("No "+lead+" data found for patient "+str(patient)+"!  Skipping...") 

#%%

# Close MySQL session
cursor.close()
session.close()
t1 = time.time()
total = t1-t0
print("\nDone.")
print("Total time elapsed :", total, "\n")
##%%
input("Press Enter to exit...")
exit(0)