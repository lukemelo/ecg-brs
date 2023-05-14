"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada NN training script
%
% Luke Melo
% Ashton Christy
% 03 Aug 2020
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#%% Import Packages
import os, pandas as pd, seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#--USER FUNCTIONS--
import nn_fns_stats_v1 as nn_stats

# %% Load experiment results

base_dir = r'Z:\Brugada_Project\ECG Paper\Results\Basale_ST_BAS_DNN_CROSSOVER'
ajm_bas_lst = ['Basale'] 
lead_strs = ['BAS','ST','BAS_ST']

      
#%%
# exclude type 1s from ajmaline cohort?
type_1_ajm_exclude = True
type_1_bas_exclude = False

# Find optimal cutoff threshold where: 
#"The best cut-off has the highest true positive rate together with the lowest false positive rate."
roc_thresh_opt = True

# preallocate array
tbl_type1_pvals = list()
# loop through basale and ajmaline results
for ajm_bas in ajm_bas_lst:
    if not('Ajmaline' in ajm_bas):
        # load reference DNN for p-value comparison
        compare_lead = lead_strs[0]
        pinfo_compare = pd.read_excel(os.path.join(base_dir,ajm_bas,compare_lead,'pinfo.xlsx'))
        holdout_reg_compare = np.asarray(pd.read_excel(os.path.join(base_dir,ajm_bas,compare_lead,'holdout_reg.xlsx')))
        rbbb_ixs = [ix for ix,phenotype in enumerate(pinfo_compare['Phenotype']) if phenotype == 'RBBB' or phenotype == 'IRBBB']
        pinfo_compare['Phenotype'][rbbb_ixs] = 'ABNORMAL'
    else:
        compare_lead = lead_strs[0]
        pinfo_compare = pd.read_excel(os.path.join(base_dir,ajm_bas,compare_lead,'pinfo.xlsx'))
        holdout_reg_compare = np.asarray(pd.read_excel(os.path.join(base_dir,ajm_bas,compare_lead,'holdout_reg.xlsx')))
        other_pinfo_dir = r'Z:\Nextcloud\Brugada_Project\Python_Scripts\BrS_CC\ECG_DATA'
        pinfo_second_read = pd.read_csv(os.path.join(other_pinfo_dir,'pinfo_ajm_quality_check.csv'),header=0, keep_default_na=False, encoding = "ISO-8859-1", engine='python')
        pinfo_second_read['Phenotype'][110] = 'NORMAL'
        pinfo_second_read['Phenotype'][153] = 'TYPE 1'
        pinfo_second_read['Phenotype'][577] = 'NORMAL'
        pinfo_second_read['Phenotype'][966] = 'IRBBB'
        ajm_SQL_ixs = [ix for ix,pnum in enumerate(pinfo_second_read['XLS Patient Number']) if pnum in np.asarray(pinfo_compare['XLS Patient Number'])]
        pinfo_second_read = pinfo_second_read.iloc[ajm_SQL_ixs].reset_index().drop(['index'],axis=1)
        pinfo_compare['XLS Patient Number'] = pinfo_second_read['XLS Patient Number']
        pinfo_compare['Gender'] = pinfo_second_read['Gender']
        pinfo_compare['Age'] = pinfo_second_read['Age']
        pinfo_compare['Phenotype'] = pinfo_second_read['Phenotype']
        #% Merge IRBBB and RBBB into abnormal ECG category
        rbbb_ixs = [ix for ix,phenotype in enumerate(pinfo_compare['Phenotype']) if phenotype == 'RBBB' or phenotype == 'IRBBB']
        pinfo_compare['Phenotype'][rbbb_ixs] = 'ABNORMAL'
        non_type1_ixs = [ix for ix,phenotype in enumerate(pinfo_compare['Phenotype']) if not(phenotype == 'TYPE 1')]
        pinfo_compare = pinfo_compare.iloc[non_type1_ixs].reset_index().drop(['index'],axis=1)
        holdout_reg_compare = holdout_reg_compare[non_type1_ixs,:]

    # loop through lead DNNs
    for lead_str in lead_strs:
        # load data
        model_dir = os.path.join(base_dir,ajm_bas,lead_str)
        pinfo = pd.read_excel(os.path.join(model_dir,'pinfo.xlsx'), sheet_name='sheet1', index=False)
        holdout_reg = np.asarray(pd.read_excel(os.path.join(model_dir,'holdout_reg.xlsx'), sheet_name='sheet1', index=False))
        
        #%% fix the pinfo problem
        if 'Ajmaline' in model_dir:
            other_pinfo_dir = r'Z:\Brugada_Project\Python_Scripts\BrS_CC\ECG_DATA'
            pinfo_second_read = pd.read_csv(os.path.join(other_pinfo_dir,'pinfo_ajm_quality_check.csv'),header=0, keep_default_na=False, encoding = "ISO-8859-1", engine='python')
            
            pinfo_second_read['Phenotype'][110] = 'NORMAL'
            pinfo_second_read['Phenotype'][153] = 'TYPE 1'
            pinfo_second_read['Phenotype'][577] = 'NORMAL'
            pinfo_second_read['Phenotype'][966] = 'IRBBB'
            
            ajm_SQL_ixs = [ix for ix,pnum in enumerate(pinfo_second_read['XLS Patient Number']) if pnum in np.asarray(pinfo['XLS Patient Number'])]
            pinfo_second_read = pinfo_second_read.iloc[ajm_SQL_ixs].reset_index().drop(['index'],axis=1)
    
            pinfo['XLS Patient Number'] = pinfo_second_read['XLS Patient Number']
            pinfo['Gender'] = pinfo_second_read['Gender']
            pinfo['Age'] = pinfo_second_read['Age']
            pinfo['Phenotype'] = pinfo_second_read['Phenotype']
            
        #%% Merge IRBBB and RBBB into abnormal ECG category
        rbbb_ixs = [ix for ix,phenotype in enumerate(pinfo['Phenotype']) if phenotype == 'RBBB' or phenotype == 'IRBBB']
        pinfo['Phenotype'][rbbb_ixs] = 'ABNORMAL'
        
        #%% Remove Type 1 for ajmaline analyis
        if 'Ajmaline' in model_dir and type_1_ajm_exclude or type_1_bas_exclude:
            non_type1_ixs = [ix for ix,phenotype in enumerate(pinfo['Phenotype']) if not(phenotype == 'TYPE 1')]
            pinfo = pinfo.iloc[non_type1_ixs].reset_index().drop(['index'],axis=1)
            holdout_reg = holdout_reg[non_type1_ixs,:]
            
        #%% Add heart rate info
        pinfo = nn_stats.add_HR_csv(pinfo,ajm_bas)
        
        #%% Initialize results table
        tbl_subsets, tbl_total_patients, tbl_accuracies, tbl_sensitivities, tbl_specificities, tbl_aucs, tbl_PPVs, tbl_NPVs, tbl_DORs, tbl_MCCs, tbl_F1s ,tbl_pvals, tbl_stds, tbl_roc_threshs = \
            list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
        
        #%% Calculate overall statistics (formulas taken from https://en.wikipedia.org/wiki/Confusion_matrix)
        # Fawcett (2006),[1] Powers (2011),[2] Ting (2011),[3] and CAWCR[4] Chicco & Jurman (2020),[5] Tharwat (2018).[6]
        
        roc_threshold, lr_auc = nn_stats.auc_roc(pinfo['Diagnosis'],holdout_reg,model_dir,roc_thresh_opt,'All')
        print('AUC-ROC: %.3f' % (lr_auc))
        tbl_aucs.append(lr_auc)
        tbl_roc_threshs.append(roc_threshold)

        TP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis]) # true positive (hit)
        TN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis]) # true negative (correct rejection)
        FN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis]) # false negative (miss, type II error)
        FP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis]) # false positive (false alarm, type I error)
        P = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1]) # condition positive (number of real positive cases)
        N = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0]) # condition nedative (number of real negative cases)
        
        TPR = 100*TP/P # sensitivity, recall, hit rate, or true positive rate (TPR)
        TNR = 100*TN/N # specificity, selectivity or true negative rate (TNR)
        PPV = 100*TP/(TP+FP) # precision or positive predictive value (PPV)
        NPV = 100*TN/(TN+FN) # negative predictive value (NPV)
        FNR = 100*FN/P # miss rate or false negative rate (FNR)
        FPR = 100*FP/N # fall-out or false positive rate (FPR)
        FDR = 100*FP/(FP+TP) # false discovery rate (FDR)
        FOR = 100*FN/(FN+TN) # false omission rate (FOR)
        PT = 100*(np.sqrt(TPR*(-TNR+1))+TNR-1)/(TPR+TNR-1) # Prevalence Threshold (PT)
        TS = 100*TP/(TP+FN+FP) # Threat score (TS) or critical success index (CSI)
        
        ACC = 100*(TP+TN)/(P+N) # accuracy (ACC)
        BA = 100*(TPR+TNR)/2 # balanced accuracy (BA)
        F1 = 2*PPV*TPR/(PPV+TPR) # F1 score (is the harmonic mean of precision and sensitivity)
        MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # Matthews correlation coefficient (MCC)
        FM = 100*np.sqrt(PPV*TPR) # Fowlkes–Mallows index (FM)
        PLR = np.divide(TPR,FPR) # positive likelihood ratio (PLR)
        NLR = np.divide(FNR,TNR) # negative likelihood ratio (NLR)
        DOR = np.divide(PLR,NLR) # diagnostics odds ratio (DOR)
        
        print('')
        print('Processing ALL: %d total patients' % (P+N))
        print('BrS: %d | Ctl: %d' % (P,N))
        print('-----------------------------------')
        print('Accuracy (ACC): %.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
        print('Sensitivity (TPR): %.1f %% (%d of %d)' % (TPR,TP,P))
        print('Specificity (TNR): %.1f %% (%d of %d)' % (TNR,TN,N))
        print('Positive Predictive Value (PPV): %.1f %% (%d of %d)' % (PPV,TP,TP+FP))
        print('Negative Predictive Value (NPV): %.1f %% (%d of %d)' % (NPV,TN,TN+FN))
        print('Diagnostic Odds Ratio (DOR): %.1f' % DOR)
        print('F1 Score (F1): %.1f' % F1)
        print('Matthews Correlation Coeffecient (MCC): %.3f' % MCC)
        
        tbl_subsets.append('ALL')
        tbl_total_patients.append(P+N)
        tbl_accuracies.append('%.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
        tbl_sensitivities.append('%.1f %% (%d of %d)' % (TPR,TP,P))
        tbl_specificities.append('%.1f %% (%d of %d)' % (TNR,TN,N))
        tbl_PPVs.append('%.1f %% (%d of %d)' % (PPV,TP,TP+FP))
        tbl_NPVs.append('%.1f %% (%d of %d)' % (NPV,TN,TN+FN))
        tbl_DORs.append(DOR)
        tbl_F1s.append(F1)
        tbl_MCCs.append(MCC)
        
        #%% Calculate <sigma> for holdout reg
        tbl_stds.append(np.nanstd(holdout_reg,axis=1))
        
        #%% p-value comparison
        pval = nn_stats.chi2_pval(pinfo_compare,holdout_reg_compare,pinfo,holdout_reg,0.95)
        tbl_pvals.append(pval)
        
        #%% Confusion Matrix
        
        conf_mtx = confusion_matrix(np.asarray(pinfo['Diagnosis']),np.round(np.asarray(np.nanmean(holdout_reg,axis=1))-roc_threshold+0.5))
        nn_stats.plot_confusion_matrix(conf_mtx,'Confusion Matrix Accuracy: ',model_dir+'\\')
        subset_ixs = [ix for ix in range(len(pinfo['Diagnosis']))]
        nn_stats.pdf_confusion_histogram_loocv(pinfo,holdout_reg,model_dir,'all',roc_threshold,subset_ixs)
        nn_stats.pdf_confusion_histogram_class_hists(pinfo,holdout_reg,model_dir,'all',roc_threshold,subset_ixs)

        print('-----------------------------------')
        print('')
        
        #%% phenotype subsets
        
        print('-----------------------------------')
        print('ECG PHENOTYPE SUBSETS')
        print('-----------------------------------')
        
        if 'Ajmaline' in model_dir and type_1_ajm_exclude or type_1_bas_exclude:
            phenotypes = ['NORMAL', 'ABNORMAL']
        else:
            phenotypes = ['NORMAL', 'ABNORMAL', 'TYPE 1']
        
        for ix,phenotype in enumerate(phenotypes):
            
            print('')
            num_phenotype = len([ix for ix,phenotype_ix in enumerate(pinfo['Phenotype']) if phenotype_ix == phenotype])
            print('Processing %s: %d total patients' % (phenotype,num_phenotype))
            tbl_subsets.append(phenotype)
            
            if not(phenotype == 'TYPE 1'):
                phenotype_ix =  [ix for ix,phenotype_ix in enumerate(pinfo['Phenotype']) if phenotype_ix == phenotype]
                holdout_reg_ph = holdout_reg[phenotype_ix]
                diagnoses = np.asarray(pinfo['Diagnosis'])[phenotype_ix]
                roc_threshold, lr_auc = nn_stats.auc_roc(diagnoses,holdout_reg_ph,model_dir,roc_thresh_opt,phenotype)
                print('AUC-ROC: %.3f' % (lr_auc))
                tbl_aucs.append(lr_auc)
                tbl_roc_threshs.append(roc_threshold)
            else:
                tbl_aucs.append(None) 
                roc_threshold = 0.5
                tbl_roc_threshs.append(roc_threshold)
            
            TP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis and pinfo['Phenotype'][ix]==phenotype]) # true positive (hit)
            TN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis and pinfo['Phenotype'][ix]==phenotype]) # true negative (correct rejection)
            FN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis and pinfo['Phenotype'][ix]==phenotype]) # false negative (miss, type II error)
            FP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis and pinfo['Phenotype'][ix]==phenotype]) # false positive (false alarm, type I error)
            P = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and pinfo['Phenotype'][ix]==phenotype]) # condition positive (number of real positive cases)
            N = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and pinfo['Phenotype'][ix]==phenotype]) # condition nedative (number of real negative cases)
            
            TPR = 100*TP/P # sensitivity, recall, hit rate, or true positive rate (TPR)
            if phenotype == 'TYPE 1':
                TNR = None
                FPR = None
                NPV = None
                FOR = None
            else:
                TNR = 100*TN/N # specificity, selectivity or true negative rate (TNR)
                FPR = 100*FP/N # fall-out or false positive rate (FPR)
                NPV = 100*TN/(TN+FN) # negative predictive value (NPV)
                FOR = 100*FN/(FN+TN) # false omission rate (FOR)
            PPV = 100*TP/(TP+FP) # precision or positive predictive value (PPV)
            FNR = 100*FN/P # miss rate or false negative rate (FNR)            
            FDR = 100*FP/(FP+TP) # false discovery rate (FDR)
            # PT = 100*(np.sqrt(TPR*(-TNR+1))+TNR-1)/(TPR+TNR-1) # Prevalence Threshold (PT)
            TS = 100*TP/(TP+FN+FP) # Threat score (TS) or critical success index (CSI)
            
            if phenotype == 'TYPE 1':
                BA = None
                NLR = None
                PLR = None
                DOR = None
            else:
                BA = 100*(TPR+TNR)/2 # balanced accuracy (BA) 
                NLR = np.divide(FNR,TNR) # negative likelihood ratio (NLR)
                PLR = np.divide(TPR,FPR) # positive likelihood ratio (PLR)
                DOR = np.divide(PLR,NLR) # diagnostics odds ratio (DOR)
                MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # Matthews correlation coefficient (MCC)
            ACC = 100*(TP+TN)/(P+N) # accuracy (ACC)
            F1 = 2*PPV*TPR/(PPV+TPR) # F1 score (is the harmonic mean of precision and sensitivity)
            FM = 100*np.sqrt(PPV*TPR) # Fowlkes–Mallows index (FM)
            
            # print('')
            # print('Processing ALL: %d total patients' % (P+N))
            print('BrS: %d | Ctl: %d' % (P,N))
            print('-----------------------------------')
            print('Accuracy (ACC): %.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
            print('Sensitivity (TPR): %.1f %% (%d of %d)' % (TPR,TP,P))
            if phenotype != 'TYPE 1':
                print('Specificity (TNR): %.1f %% (%d of %d)' % (TNR,TN,N))
                print('Positive Predictive Value (PPV): %.1f %% (%d of %d)' % (PPV,TP,TP+FP))
                print('Negative Predictive Value (NPV): %.1f %% (%d of %d)' % (NPV,TN,TN+FN))
                print('Diagnostic Odds Ratio (DOR): %.1f' % DOR)
                print('Matthews Correlation Coeffecient (MCC): %.3f' % MCC)
            print('F1 Score (F1): %.1f' % F1)
            
            tbl_total_patients.append(P+N)
            tbl_accuracies.append('%.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
            tbl_sensitivities.append('%.1f %% (%d of %d)' % (TPR,TP,P))
            if phenotype != 'TYPE 1':
                tbl_specificities.append('%.1f %% (%d of %d)' % (TNR,TN,N))
                tbl_PPVs.append('%.1f %% (%d of %d)' % (PPV,TP,TP+FP))
                tbl_NPVs.append('%.1f %% (%d of %d)' % (NPV,TN,TN+FN))
                tbl_DORs.append(DOR)
                tbl_MCCs.append(MCC)
            else:
                tbl_specificities.append(None)
                tbl_PPVs.append(None)
                tbl_NPVs.append(None)
                tbl_DORs.append(None)
                tbl_MCCs.append(None)
            tbl_F1s.append(F1)
            
            #%% Calculate p-value
            pval = nn_stats.chi2_pval_subsets(pinfo,holdout_reg,'Phenotype','NORMAL',phenotype,0.95)
            tbl_pvals.append(pval)
            if phenotype == 'TYPE 1':
                pval_type1 = nn_stats.chi2_pval_type1(pinfo_compare,holdout_reg_compare,pinfo,holdout_reg,0.95)
                tbl_type1_pvals.append(pval_type1)
            
            #%% Confusion Matrix
            phenotype_ix =  [ix for ix,phenotype_ix in enumerate(pinfo['Phenotype']) if phenotype_ix == phenotype]
            
            nn_stats.pdf_confusion_histogram_loocv(pinfo,holdout_reg,model_dir,phenotype,roc_threshold,phenotype_ix)
            tbl_stds.append(np.nanstd(holdout_reg[phenotype_ix],axis=1))

            if not(phenotype == 'TYPE 1'):
                conf_mtx = confusion_matrix(diagnoses,np.round(np.asarray(np.nanmean(holdout_reg_ph,axis=1))-roc_threshold+0.5))
                nn_stats.plot_confusion_matrix(conf_mtx,phenotype +' | Confusion Matrix Accuracy: ',model_dir+'\\'+phenotype+'_')
            else:
                pinfo_t1 = pinfo
  
            print('-----------------------------------')
            print('')

        #%% gender subsets
        
        print('-----------------------------------')
        print('GENDER SUBSETS')
        print('-----------------------------------')
        
        genders = ['M','F']
        
        for ix,gender in enumerate(genders):
            print('')
            tbl_subsets.append(gender)
            num_gender = len([ix for ix,gender_ix in enumerate(pinfo['Gender']) if gender_ix == gender])
            print('Processing %s: %d total patients' % (gender,num_gender))

            # ROC
            gender_ix =  [ix for ix,gender_ix in enumerate(pinfo['Gender']) if gender_ix == gender]
            holdout_reg_ph = holdout_reg[gender_ix]
            diagnoses = np.asarray(pinfo['Diagnosis'])[gender_ix]
            roc_threshold, lr_auc = nn_stats.auc_roc(diagnoses,holdout_reg_ph,model_dir,roc_thresh_opt,gender)
            print('AUC-ROC: %.3f' % (lr_auc))
            tbl_aucs.append(lr_auc)
            tbl_roc_threshs.append(roc_threshold)
            
            TP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis and pinfo['Gender'][ix]==gender]) # true positive (hit)
            TN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) == diagnosis and pinfo['Gender'][ix]==gender]) # true negative (correct rejection)
            FN = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis and pinfo['Gender'][ix]==gender]) # false negative (miss, type II error)
            FP = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])-roc_threshold+0.5) != diagnosis and pinfo['Gender'][ix]==gender]) # false positive (false alarm, type I error)
            P = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and pinfo['Gender'][ix]==gender]) # condition positive (number of real positive cases)
            N = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and pinfo['Gender'][ix]==gender]) # condition nedative (number of real negative cases)
            
            TPR = 100*TP/P # sensitivity, recall, hit rate, or true positive rate (TPR)
            TNR = 100*TN/N # specificity, selectivity or true negative rate (TNR)
            PPV = 100*TP/(TP+FP) # precision or positive predictive value (PPV)
            NPV = 100*TN/(TN+FN) # negative predictive value (NPV)
            FNR = 100*FN/P # miss rate or false negative rate (FNR)
            FPR = 100*FP/N # fall-out or false positive rate (FPR)
            FDR = 100*FP/(FP+TP) # false discovery rate (FDR)
            FOR = 100*FN/(FN+TN) # false omission rate (FOR)
            # PT = 100*(np.sqrt(TPR*(-TNR+1))+TNR-1)/(TPR+TNR-1) # Prevalence Threshold (PT)
            TS = 100*TP/(TP+FN+FP) # Threat score (TS) or critical success index (CSI)
            
            ACC = 100*(TP+TN)/(P+N) # accuracy (ACC)
            BA = 100*(TPR+TNR)/2 # balanced accuracy (BA)
            F1 = 2*PPV*TPR/(PPV+TPR) # F1 score (is the harmonic mean of precision and sensitivity)
            MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) # Matthews correlation coefficient (MCC)
            FM = 100*np.sqrt(PPV*TPR) # Fowlkes–Mallows index (FM)
            PLR = np.divide(TPR,FPR) # positive likelihood ratio (PLR)
            NLR = np.divide(FNR,TNR) # negative likelihood ratio (NLR)
            DOR = np.divide(PLR,NLR) # diagnostics odds ratio (DOR)
            
            print('BrS: %d | Ctl: %d' % (P,N))
            print('-----------------------------------')
            print('Accuracy (ACC): %.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
            print('Sensitivity (TPR): %.1f %% (%d of %d)' % (TPR,TP,P))
            print('Specificity (TNR): %.1f %% (%d of %d)' % (TNR,TN,N))
            print('Positive Predictive Value (PPV): %.1f %% (%d of %d)' % (PPV,TP,TP+FP))
            print('Negative Predictive Value (NPV): %.1f %% (%d of %d)' % (NPV,TN,TN+FN))
            print('Diagnostic Odds Ratio (DOR): %.1f' % DOR)
            print('F1 Score (F1): %.1f' % F1)
            print('Matthews Correlation Coeffecient (MCC): %.3f' % MCC)

            tbl_total_patients.append(P+N)
            tbl_accuracies.append('%.1f %% (%d of %d)' % (ACC,TP+TN,P+N))
            tbl_sensitivities.append('%.1f %% (%d of %d)' % (TPR,TP,P))
            tbl_specificities.append('%.1f %% (%d of %d)' % (TNR,TN,N))
            tbl_PPVs.append('%.1f %% (%d of %d)' % (PPV,TP,TP+FP))
            tbl_NPVs.append('%.1f %% (%d of %d)' % (NPV,TN,TN+FN))
            tbl_DORs.append(DOR)
            tbl_F1s.append(F1)
            tbl_MCCs.append(MCC)
            
            #%% Calculate p-value
            pval = nn_stats.chi2_pval_subsets(pinfo,holdout_reg,'Gender','M',gender,0.95)
            tbl_pvals.append(pval)
            
            #%% Confusion Matrix
        
            gender_ix =  [ix for ix,gender_ix in enumerate(pinfo['Gender']) if gender_ix == gender]
            conf_mtx = confusion_matrix(diagnoses,np.round(np.asarray(np.nanmean(holdout_reg_ph,axis=1))-roc_threshold+0.5))
            nn_stats.plot_confusion_matrix(conf_mtx,phenotype +' | Confusion Matrix Accuracy: ',model_dir+'\\'+gender+'_')
            nn_stats.pdf_confusion_histogram_loocv(pinfo,holdout_reg,model_dir,gender,roc_threshold,gender_ix)
            
            tbl_stds.append(np.nanstd(holdout_reg[gender_ix],axis=1))

            print('-----------------------------------')
            print('')

        #%% Build table with subset specific statistics
        

        table_statistics = pd.DataFrame({"Patient Subgroup":tbl_subsets,\
                                  "Total Patients":tbl_total_patients,\
                                  "Sensitivity":tbl_sensitivities,\
                                  "Specificity":tbl_specificities,\
                                  "AUC-ROC":tbl_aucs,\
                                  "ROC Threshold":tbl_roc_threshs,\
                                  "Accuracy":tbl_accuracies,\
                                  "Positive Predictive Value":tbl_PPVs,\
                                  "Negative Predictive Value":tbl_NPVs,\
                                  "Diagnostic Odds Ratio":tbl_DORs,\
                                  "F1 Score":tbl_F1s,\
                                  "MCC":tbl_MCCs,\
                                  "p-value":tbl_pvals}) 
        table_statistics.to_excel(os.path.join(model_dir,'subset_statistics.xlsx'), sheet_name='sheet1', index=False)
        
        
        #%% Build table for stdev plots
        labels,stds = list(),list()
        for ix,subset in enumerate(tbl_subsets):
            labels += [subset]*len(tbl_stds[ix])
            stds += list(tbl_stds[ix]) 
        
        
        table_stds = pd.DataFrame({"Patient Subgroup":labels,\
                                  "Regression Stds":stds})
        fig = plt.figure(figsize=(6,6))    
        ax = sns.boxplot(x="Regression Stds", y="Patient Subgroup",
                            data=table_stds,
                            # scale="width", 
                            palette="Set3",
                            showfliers = False)
        plt.title('Regression Standard Deviations')
        ax.set_xlabel('Regression DNN Standard Deviation')
        plt.rc('font', size=8)          # controls default text sizes
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
        plt.rc('legend', fontsize=8)    # legend fontsize
        plt.rc('figure', titlesize=10)  # fontsize of the figure title
        # ax.set(xlim=(0,0.4))
        fig = ax.get_figure()
        fig.savefig(os.path.join(model_dir,'regression_stdev_breakdown.pdf'),dpi=300,bbox_inches='tight')
        ax.set(xlim=(0,0.4))
        fig.savefig(os.path.join(model_dir,'regression_stdev_breakdown_scaled.pdf'),dpi=300,bbox_inches='tight')
        