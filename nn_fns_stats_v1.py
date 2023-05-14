"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada NN statistical analysis functions
%
% Luke Melo
% Ashton Christy
% 23 December 2020
% 
% Release Notes:
%   - Replacing NN_fns_luke specifically for stats funcitons
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
# Import standard modules
import pandas as pd, numpy as np, mysql.connector, os, matplotlib.pyplot as plt, scipy.optimize
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

#%% AUC-ROC Functions

def auc_roc(diagnoses,holdout_reg,model_dir,roc_thresh_opt,subset):
    """
    Generates an AUC-ROC curve
    ----------
    Inputs:
        diagnoses-- list of true diagnoses for the patients
        holdout_reg-- 2D array of k-folv CV DNN scores for each patient
        model_dir-- directory to save data
        roc_thresh_opt-- [Boolean] maximize PPV for cutoff value if true, else 0.5
        subset-- specific subset of the data used
        
    Outputs:
        roc_threshold-- roc threshold value calculated or 0.5
        lr_auc-- logistic regression area under curve (AUC value)
        
    ----------
    """
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(holdout_reg.shape[0])]
    ns_fpr, ns_tpr, _ = roc_curve(diagnoses, ns_probs)
    # predict probabilities
    lr_probs = np.asarray(np.nanmean(holdout_reg,axis=1))
    # AUC value
    lr_auc = roc_auc_score(diagnoses, lr_probs)
    # calculate roc curves
    lr_fpr, lr_tpr, thresholds = roc_curve(diagnoses, lr_probs)
    # calculate optimal threshold for ROC
    if not(roc_thresh_opt):
        roc_threshold = 0.5
        optimal_idx = (np.abs(thresholds-roc_threshold)).argmin()
    else:
        optimal_idx = np.argmax(lr_tpr - lr_fpr)
        roc_threshold = thresholds[optimal_idx]
    # plot the roc curve for the model
    plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.plot(lr_fpr[optimal_idx], lr_tpr[optimal_idx], marker='.', label='Threshold')
    # axis labels
    plt.title('%s | ROC AUC=%.3f | Threshold=%.3f' % (subset,lr_auc,roc_threshold), fontsize=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # show the legend
    plt.legend()
    # save
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir,'auc_roc_%s.pdf' % subset))
    # show the plot
    plt.show()
    return roc_threshold,lr_auc

#%% Confustion Matrix / Histogram Functions

def plot_confusion_matrix(cm, title, save_dir, cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    ----------
    Inputs:
        cm-- p2 by 2 array confusion matrix
        title-- plot title
        save_dir: directory to save pdf
        
    Outputs:
        acc-- accuracy of the model
    ----------
    """
    import itertools
    plt.figure(figsize=(6,6))
    np.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.xticks([0, 1], ['Negative','Positive'], rotation=45)
    plt.yticks([0, 1], ['Negative','Positive'])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm_pct.max() / 2.
    for ii,jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(jj,ii, '{:,.2%}'.format(cm_pct[ii,jj])+" ("+str(cm[ii,jj])+")", \
                 horizontalalignment="center", color="white" \
                 if cm_pct[ii,jj] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True diagnosis')
    plt.xlabel('Predicted diagnosis')
    plt.title(title+str('{:,.2%}'.format(np.trace(cm)/np.sum(cm))) ,fontsize=10)
    plt.savefig(save_dir + 'confusion_mtx.pdf')
    acc = np.trace(cm)/np.sum(cm)
    return acc

def parabola(x, a, b, c):
    # simple parabola function
    return a*x**2 + b*x + c

def pdf_confusion_histogram_loocv(pinfo,holdout_reg,save_path,subcategory,roc_threshold,subset_ixs):
    """
    Plot confusion histogram for a leave one out cross validation dataset
    ----------
    Inputs:
        pinfo: patient info
        holdout_reg: regression output values for each patient
        save_path: directory to save pdf
        subcategory: subcategory of patients to label pdf
        roc_threshold: cutoff value for DNN classification
        subset ixs: list of indicies of patients for subset
        
    Outputs:
        None
    ----------
    """
    # get patient subsets info
    diagnoses = np.asarray(pinfo['Diagnosis'][subset_ixs])
    HRs = np.asarray(pinfo['HR'][subset_ixs])
    HRVs = np.asarray(pinfo['HRV'][subset_ixs])
    holdout_reg = holdout_reg[subset_ixs]
    
    # calculate the meanregression score for each patient
    avg_scores = np.nanmean(holdout_reg,axis=1)
    # diagnoses = np.asarray(pinfo['Diagnosis'])
    
    # compare average regression value to their true diagnosis
    true_regs = np.asarray([score for ix,score in enumerate(avg_scores) if np.round(score-roc_threshold+0.5) == diagnoses[ix]])

    # plot and save histogram
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram | %s' % subcategory,fontsize=10)  
    plt.hist(100*avg_scores,bins=range(0,110,10),edgecolor='k',histtype='bar',stacked=True, color=['#1f77b4'])
    plt.hist(100*true_regs,bins=range(0,110,10),edgecolor='k',histtype='bar',stacked=True, color=['#ff7f0e'])
    plt.xlim(0,100)
    # plt.xlim(0,1)
    plt.xticks(np.linspace(0,100,6),[str(x/100) for x in range(0,110,20)])
    plt.xlabel('DNN Output Score')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_%s.pdf' % subcategory))
    
    #standard deviation scores for each decade
    dnn_decades, stdevs_avgs, stdevs_stds, HR_avgs, HR_stds, HRV_avgs, HRV_stds = \
        np.linspace(0.05,0.95,10),list(),list(),list(),list(),list(),list()
    for val in dnn_decades:
        holdout_reg_ixs = [ix for ix,reg_avg in enumerate(holdout_reg) if val - 0.05 <= np.nanmean(reg_avg) <= val + 0.05]
        stdevs_avgs.append(np.nanmean(np.nanstd(holdout_reg[holdout_reg_ixs],axis=1)))
        stdevs_stds.append(np.nanstd(np.nanstd(holdout_reg[holdout_reg_ixs],axis=1)))
        HR_avgs.append(np.nanmean(HRs[holdout_reg_ixs]))
        HR_stds.append(np.nanstd(HRs[holdout_reg_ixs]))
        HRV_avgs.append(np.nanmean(HRVs[holdout_reg_ixs]))
        HRV_stds.append(np.nanstd(HRVs[holdout_reg_ixs]))
        
    # plot and save histogram
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram of Average Stdevs | %s' % subcategory,fontsize=10) 
    # plt.scatter(dnn_decades,stdevs_avgs)
    plt.errorbar(dnn_decades, stdevs_avgs, yerr=stdevs_stds, fmt='o')
    for _ in [1]:
        try:
            fit_params, pcov = scipy.optimize.curve_fit(parabola, dnn_decades, stdevs_avgs)
            y_fit = parabola(dnn_decades, *fit_params)
            plt.plot(dnn_decades, y_fit, label='fit')
        except:
            continue
    plt.xlim(0,1)
    plt.xlabel('DNN Output Score')
    plt.ylabel('Average Standard Deviation')    
    plt.xticks(np.linspace(0,1,6),[str(x/100) for x in range(0,110,20)])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_avg_stdevs_%s.pdf' % subcategory))
    
    # plot and save HR histogram
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram of Heart Rates | %s' % subcategory,fontsize=10) 
    # plt.scatter(dnn_decades,stdevs_avgs)
    plt.errorbar(dnn_decades, HR_avgs, yerr=HR_stds, fmt='-o')
    plt.xlim(0,1)
    plt.xlabel('DNN Output Score')
    plt.ylabel('Average Heart Rate')    
    plt.xticks(np.linspace(0,1,6),[str(x/100) for x in range(0,110,20)])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_avg_HRs_%s.pdf' % subcategory))
    
    # plot and save histogram
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram of Heart Rate Variabilities | %s' % subcategory,fontsize=10) 
    # plt.scatter(dnn_decades,stdevs_avgs)
    plt.errorbar(dnn_decades, HRV_avgs, yerr=HRV_stds, fmt='-o')
    plt.xlim(0,1)
    plt.xlabel('DNN Output Score')
    plt.ylabel('HRV (Percent stdev)')    
    plt.xticks(np.linspace(0,1,6),[str(x/100) for x in range(0,110,20)])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_avg_HRVs_%s.pdf' % subcategory))
    
def pdf_confusion_histogram_class_hists(pinfo,holdout_reg,save_path,subcategory,roc_threshold,subset_ixs):
    """
    Plot confusion histogram for a leave one out cross validation dataset
    ----------
    Inputs:
        pinfo: patient info
        holdout_reg: regression output values for each patient
        save_path: directory to save pdf
        subcategory: subcategory of patients to label pdf
        roc_threshold: cutoff value for DNN classification
        subset ixs: list of indicies of patients for subset
        
    Outputs:
        None
    ----------
    """
    # get patient subsets info
    diagnoses = np.asarray(pinfo['Diagnosis'][subset_ixs])
    holdout_reg = holdout_reg[subset_ixs]
    
    # calculate the mean regression score for each patient
    avg_scores = np.nanmean(holdout_reg,axis=1)
    
    pos_regs = np.asarray([score for ix,score in enumerate(avg_scores) if 1 == diagnoses[ix]])
    ctl_regs = np.asarray([score for ix,score in enumerate(avg_scores) if 0 == diagnoses[ix]])
    

    # plot and save histogram for positive patients
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram | Positive | %s' % subcategory,fontsize=10) 
    plt.hist(100*pos_regs,bins=range(0,110,10),edgecolor='k',histtype='bar',stacked=True, color=['#d62728'])
    plt.xlim(0,100)
    # plt.xlim(0,1)
    plt.xticks(np.linspace(0,100,6),[str(x/100) for x in range(0,110,20)])
    plt.xlabel('DNN Output Score')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_pos_%s.pdf' % subcategory))
    
    # plot and save histogram for control patients
    fig=plt.figure(figsize=(6,6))
    plt.rc('font', size=8)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.title('Confusion Histogram | Control | %s' % subcategory,fontsize=10)  
    plt.hist(100*ctl_regs,bins=range(0,110,10),edgecolor='k',histtype='bar',stacked=True, color=['#2ca02c'])
    plt.xlim(0,100)
    # plt.xlim(0,1)
    plt.xticks(np.linspace(0,100,6),[str(x/100) for x in range(0,110,20)])
    plt.xlabel('DNN Output Score')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'confusion_histogram_ctl_%s.pdf' % subcategory))


#%% Chi2 Functions

def chi2_pval(pinfo_1,holdout_reg_1,pinfo_2,holdout_reg_2,prob):
    """
    Calculates the p-value for the chi2 distribution comparing two DNNs accuracy
    
    Input:
        pinfo_1, pinfo_2: pinfo dataframes from DNN output to compare
        holdout_reg_1, holdout_reg_2: numpy arrays with k-fold DNN scores for each patient
        prob: significance level of the p-value test (typical 0.95)
        
    Output:
        p: p-value obtained from the chi2 test for significance
    """

    # Calculate accuracy contingency table parameters for both DNNs
    TP_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_1[ix,:])) == diagnosis]) # true positive (hit)
    TN_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_1[ix,:])) == diagnosis]) # true negative (correct rejection)
    FN_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_1[ix,:])) != diagnosis]) # false negative (miss, type II error)
    FP_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_1[ix,:])) != diagnosis]) # false positive (false alarm, type I error)

    TP_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_2[ix,:])) == diagnosis]) # true positive (hit)
    TN_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_2[ix,:])) == diagnosis]) # true negative (correct rejection)
    FN_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_2[ix,:])) != diagnosis]) # false negative (miss, type II error)
    FP_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_2[ix,:])) != diagnosis]) # false positive (false alarm, type I error)

    #    DNN   | Correctly Classified  | Incorrectly Classified
    #     1    |        TN + TP        |        FN + FP 
    #     2    |        TN + TP        |        FN + FP 
    table = [[TP_1 + TN_1,FP_1 + FN_1],
             [TP_2 + TN_2,FP_2 + FN_2]]
      
    # calculate chi2 and interpret test-statistic
    stat, p, dof, expected = stats.chi2_contingency(table,correction=False) # turn off yates correction

    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Dependent (reject H0)')
    else:
    	print('Independent (fail to reject H0)')
    
    return p

def chi2_pval_type1(pinfo_1,holdout_reg_1,pinfo_2,holdout_reg_2,prob):
    """
    Calculates the p-value for the chi2 distribution comparing two DNNs accuracy
    
    Input:
        pinfo_1, pinfo_2: pinfo dataframes from DNN output to compare
        holdout_reg_1, holdout_reg_2: numpy arrays with k-fold DNN scores for each patient
        prob: significance level of the p-value test (typical 0.95)
        
    Output:
        p: p-value obtained from the chi2 test for significance
    """

    # Calculate accuracy contingency table parameters for both DNNs
    TP_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_1[ix,:])) == diagnosis and pinfo_1['Phenotype'][ix]=='TYPE 1']) # true positive (hit)
    TN_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_1[ix,:])) == diagnosis and pinfo_1['Phenotype'][ix]=='TYPE 1']) # true negative (correct rejection)
    FN_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_1[ix,:])) != diagnosis and pinfo_1['Phenotype'][ix]=='TYPE 1']) # false negative (miss, type II error)
    FP_1 = len([1 for ix,diagnosis in enumerate(pinfo_1['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_1[ix,:])) != diagnosis and pinfo_1['Phenotype'][ix]=='TYPE 1']) # false positive (false alarm, type I error)

    TP_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_2[ix,:])) == diagnosis and pinfo_2['Phenotype'][ix]=='TYPE 1']) # true positive (hit)
    TN_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_2[ix,:])) == diagnosis and pinfo_2['Phenotype'][ix]=='TYPE 1']) # true negative (correct rejection)
    FN_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg_2[ix,:])) != diagnosis and pinfo_2['Phenotype'][ix]=='TYPE 1']) # false negative (miss, type II error)
    FP_2 = len([1 for ix,diagnosis in enumerate(pinfo_2['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg_2[ix,:])) != diagnosis and pinfo_2['Phenotype'][ix]=='TYPE 1']) # false positive (false alarm, type I error)

    #    DNN   | Correctly Classified  | Incorrectly Classified
    #     1    |        TN + TP        |        FN + FP 
    #     2    |        TN + TP        |        FN + FP 
    table = [[TP_1 + TN_1,FP_1 + FN_1],
             [TP_2 + TN_2,FP_2 + FN_2]]
      
    # calculate chi2 and interpret test-statistic
    stat, p, dof, expected = stats.chi2_contingency(table,correction=False) # turn off yates correction

    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Dependent (reject H0)')
    else:
    	print('Independent (fail to reject H0)')
    
    return p

def chi2_pval_subsets(pinfo,holdout_reg,subset,subset_base,subset_compare,prob):
    """
    Calculates the p-value for the chi2 distribution within a DNN
    
    Input:
        pinfo_1, pinfo_2: pinfo dataframes from DNN output to compare
        holdout_reg_1, holdout_reg_2: numpy arrays with k-fold DNN scores for each patient
        prob: significance level of the p-value test (typical 0.95)
        
    Output:
        p: p-value obtained from the chi2 test for significance
    """
    TP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and pinfo[subset][ix]==subset_base]) # true positive (hit)
    TN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and pinfo[subset][ix]==subset_base]) # true negative (correct rejection)
    FN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and pinfo[subset][ix]==subset_base]) # false negative (miss, type II error)
    FP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and pinfo[subset][ix]==subset_base]) # false positive (false alarm, type I error)

    TP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and pinfo[subset][ix]==subset_compare]) # true positive (hit)
    TN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and pinfo[subset][ix]==subset_compare]) # true negative (correct rejection)
    FN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and pinfo[subset][ix]==subset_compare]) # false negative (miss, type II error)
    FP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and pinfo[subset][ix]==subset_compare]) # false positive (false alarm, type I error)

    #    DNN   | Correctly Classified  | Incorrectly Classified
    #     1    |        TN + TP        |        FN + FP 
    #     2    |        TN + TP        |        FN + FP 
    table = [[TP_1 + TN_1,FP_1 + FN_1],
             [TP_2 + TN_2,FP_2 + FN_2]]
      
    # calculate chi2 and interpret test-statistic
    stat, p, dof, expected = stats.chi2_contingency(table,correction=False) # turn off yates correction

    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Dependent (reject H0)')
    else:
    	print('Independent (fail to reject H0)')
    
    return p

def chi2_pval_subsets_hr(pinfo,holdout_reg,subset_base,subset_compare,prob):
    """
    Calculates the p-value for the chi2 distribution within a DNN
    
    Input:
        pinfo_1, pinfo_2: pinfo dataframes from DNN output to compare
        holdout_reg_1, holdout_reg_2: numpy arrays with k-fold DNN scores for each patient
        prob: significance level of the p-value test (typical 0.95)
        
    Output:
        p: p-value obtained from the chi2 test for significance
    """
    # subset = 'Age'
    hr_ranges = ['hr_0-50','hr_50-60','hr_60-70','hr_70-80','hr_80-90','hr_90-100','hr_100+','hr_50-100']
    hr_lim_lower = [0 ,50,60,70,80,90,100,50]
    hr_lim_upper = [50,60,70,80,90,100,1000,100]
    
    base_ix = [ix for ix,hr_range in enumerate(hr_ranges) if hr_range == subset_base][0]
    compare_ix = [ix for ix,hr_range in enumerate(hr_ranges) if hr_range == subset_compare][0]
    
    TP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and hr_lim_lower[base_ix] <= pinfo['HR'][ix] < hr_lim_upper[base_ix]]) # true positive (hit)
    TN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and hr_lim_lower[base_ix] <= pinfo['HR'][ix] < hr_lim_upper[base_ix]]) # true negative (correct rejection)
    FN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and hr_lim_lower[base_ix] <= pinfo['HR'][ix] < hr_lim_upper[base_ix]]) # false negative (miss, type II error)
    FP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and hr_lim_lower[base_ix] <= pinfo['HR'][ix] < hr_lim_upper[base_ix]]) # false positive (false alarm, type I error)
            
    TP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and hr_lim_lower[compare_ix] <= pinfo['HR'][ix] < hr_lim_upper[compare_ix]]) # true positive (hit)
    TN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and hr_lim_lower[compare_ix] <= pinfo['HR'][ix] < hr_lim_upper[compare_ix]]) # true negative (correct rejection)
    FN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and hr_lim_lower[compare_ix] <= pinfo['HR'][ix] < hr_lim_upper[compare_ix]]) # false negative (miss, type II error)
    FP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and hr_lim_lower[compare_ix] <= pinfo['HR'][ix] < hr_lim_upper[compare_ix]]) # false positive (false alarm, type I error)
    
    #    DNN   | Correctly Classified  | Incorrectly Classified
    #     1    |        TN + TP        |        FN + FP 
    #     2    |        TN + TP        |        FN + FP 
    table = [[TP_1 + TN_1,FP_1 + FN_1],
             [TP_2 + TN_2,FP_2 + FN_2]]
      
    # calculate chi2 and interpret test-statistic
    stat, p, dof, expected = stats.chi2_contingency(table,correction=False) # turn off yates correction

    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Dependent (reject H0)')
    else:
    	print('Independent (fail to reject H0)')
    
    return p

def chi2_pval_subsets_age(pinfo,holdout_reg,subset_base,subset_compare,prob):
    """
    Calculates the p-value for the chi2 distribution within a DNN
    
    Input:
        pinfo_1, pinfo_2: pinfo dataframes from DNN output to compare
        holdout_reg_1, holdout_reg_2: numpy arrays with k-fold DNN scores for each patient
        prob: significance level of the p-value test (typical 0.95)
        
    Output:
        p: p-value obtained from the chi2 test for significance
    """
    subset = 'Age'
    age_ranges = ['0-20','20-30','30-40','40-50','50-60','60+'] 
    age_lim_lower = [0 ,20,30,40,50,60]
    age_lim_upper = [20,30,40,50,60,1000]
    
    base_ix = [ix for ix,age_range in enumerate(age_ranges) if age_range == subset_base][0]
    compare_ix = [ix for ix,age_range in enumerate(age_ranges) if age_range == subset_compare][0]
    
    TP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and age_lim_lower[base_ix] <= pinfo['Age'][ix] < age_lim_upper[base_ix]]) # true positive (hit)
    TN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and age_lim_lower[base_ix] <= pinfo['Age'][ix] < age_lim_upper[base_ix]]) # true negative (correct rejection)
    FN_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and age_lim_lower[base_ix] <= pinfo['Age'][ix] < age_lim_upper[base_ix]]) # false negative (miss, type II error)
    FP_1 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and age_lim_lower[base_ix] <= pinfo['Age'][ix] < age_lim_upper[base_ix]]) # false positive (false alarm, type I error)
            
    TP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and age_lim_lower[compare_ix] <= pinfo['Age'][ix] < age_lim_upper[compare_ix]]) # true positive (hit)
    TN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) == diagnosis and age_lim_lower[compare_ix] <= pinfo['Age'][ix] < age_lim_upper[compare_ix]]) # true negative (correct rejection)
    FN_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 1 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and age_lim_lower[compare_ix] <= pinfo['Age'][ix] < age_lim_upper[compare_ix]]) # false negative (miss, type II error)
    FP_2 = len([1 for ix,diagnosis in enumerate(pinfo['Diagnosis']) if diagnosis == 0 and np.round(np.nanmean(holdout_reg[ix,:])) != diagnosis and age_lim_lower[compare_ix] <= pinfo['Age'][ix] < age_lim_upper[compare_ix]]) # false positive (false alarm, type I error)
    
    #    DNN   | Correctly Classified  | Incorrectly Classified
    #     1    |        TN + TP        |        FN + FP 
    #     2    |        TN + TP        |        FN + FP 
    table = [[TP_1 + TN_1,FP_1 + FN_1],
             [TP_2 + TN_2,FP_2 + FN_2]]
      
    # calculate chi2 and interpret test-statistic
    stat, p, dof, expected = stats.chi2_contingency(table,correction=False) # turn off yates correction

    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
    	print('Dependent (reject H0)')
    else:
    	print('Independent (fail to reject H0)')
    
    return p

def remove_kfold_std(pinfo,holdout_reg,std_thresh):
    """
    remove patients with high dnn output score variability from dataset
    """
    # isolate outliers with high standard deviatoions in dnn output score
    reg_outliers_ixs = [ix for ix,holdout_std in enumerate(np.std(holdout_reg,axis=1)) if holdout_std <= std_thresh]
    # adjust pinfo to remove outliers and reset from original indexing
    pinfo = pinfo.iloc[reg_outliers_ixs].reset_index().drop(['index'],axis=1)
    holdout_reg = holdout_reg[reg_outliers_ixs]
    return pinfo,holdout_reg

#%% Data fetching from SQL

def add_HR(pinfo,ajm_bas):
    """
    Add heart rate information to the patient info dataframe from SQL
    """
    # Connect to SQL
    session = mysql.connector.connect(host='localhost',database='brugada',user='root',password='root',use_pure=True)
    cursor = session.cursor(buffered=True)
    # loop through SQL fields of interest to extract patient info into dataframe
    if 'bas' in ajm_bas.lower():
        hr_field = 'Baseline_HR'
    elif 'ajm' in ajm_bas.lower():
        hr_field = 'Ajmaline_HR'
    SQL_fields = ['PatientID',hr_field]
    pinfo_SQL = pd.DataFrame(columns=SQL_fields)
    for field_ix,field in enumerate(SQL_fields):
        sql_fetch_patients = """SELECT %s FROM `patient_info`""" % field
        cursor.execute(sql_fetch_patients)
        pinfo_SQL[field] = [ii[0] for ii in cursor.fetchall()]
    # find overlapping patients and insert into pinfo
    try:
        SQL_ixs = [ix for ix,pID in enumerate(pinfo_SQL['PatientID']) if pID in np.asarray(pinfo['XLS Patient Number'])]
    except:
        SQL_ixs = [ix for ix,pID in enumerate(pinfo_SQL['PatientID']) if pID in np.asarray(pinfo['PatientID'])]  
    pinfo_SQL = pinfo_SQL.iloc[SQL_ixs].reset_index().drop(['index'],axis=1)
    pinfo['HR'] = np.asarray(pinfo_SQL[hr_field])
    # Add HRV info for each patient
    hrv = list()
    try:
        patients = np.asarray(pinfo['XLS Patient Number'])
        print(patients)
    except:
        patients = np.asarray(pinfo['PatientID'])
    for rr,patient_id in enumerate(patients):
        try:
            if 'bas' in ajm_bas.lower():
                sql_fetch = """SELECT QRS_peaks FROM `baseline_traces_chopped` WHERE `patientid` = """+str(patient_id)+""
            elif 'ajm' in ajm_bas.lower():
                sql_fetch = """SELECT QRS_peaks FROM `ajmaline_traces_chopped` WHERE `patientid` = """+str(patient_id)+""
            cursor.execute(sql_fetch)
            QRS_peaks = cursor.fetchall()
            QRS_peaks = np.frombuffer(QRS_peaks[0][0], dtype='int')
            QRS_med = np.median(np.diff(QRS_peaks))
            QRS_std = np.nanstd(np.diff(QRS_peaks))
            row = [patient_id, 120000/QRS_med, QRS_std/QRS_med]
            hrv.append(row)
        except:
            continue
    print('Here')
    hrv = [hrv_ix[2] for ix,hrv_ix in enumerate(hrv)]
    print('%s cohort with %d hrvs' % (ajm_bas,len(hrv)))
    pinfo['HRV'] = hrv
    
    return pinfo

def add_HR_csv(pinfo,ajm_bas):
    """
    Add heart rate information to the patient info dataframe from csv file
    """
    # load HR and HRV data for cohort
    hr_csv = os.path.join(r'Z:\Nextcloud\Brugada_Project\Python_Scripts\BrS_CC\ECG_DATA','hrv_%s.csv' % ajm_bas.lower())
    df_HR = np.asarray(pd.read_csv(hr_csv,header=None))
    print(hr_csv)
    # filter out for pIDs needed
    pID_ixs = [ix for ix,pID in enumerate(pinfo['XLS Patient Number']) if pID in df_HR[:,0]]
    # pID_ixs = [ix for ix,pID in enumerate(df_HR[:,0]) if int(pID) in np.asarray(pinfo['XLS Patient Number'])]
    df_HR = df_HR[pID_ixs]
    # insert HR and HRV data into pinfo dataframe
    pinfo['HR'] = df_HR[:,1]
    pinfo['HRV'] = df_HR[:,2]
    return pinfo
    
    
    
    