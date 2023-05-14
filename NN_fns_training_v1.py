"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada NN training functions
%
% Luke Melo
% Ashton Christy
% 23 December 2020
% 
% Release Notes:
%   - Replacing NN_fns_luke specifically for training funcitons
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Import standard modules
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle, os
from scipy import signal

# Import sklearn modules
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Import keras modules
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, GaussianNoise
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from keras import backend as K

# SQL
import mysql.connector

#%% Database Loading Functions

def load_ECG_cohort(cohort,use_leads,pinfo_csv,remove_missing_phenotype):
    """
    Loads ECG database cohort directly from the SQL server
    ----------
    Inputs:
        cohort: either 'Basale' or 'Ajmaline' to identify the ECG cohort to use
        use_leads: list of leads to pull from database
        pinfo_csv: filepath of pinfo csv file
        remove_missing_phenotype: remove patients with no phenotype info?
        
    Outputs:
        X: 2D matrix of ECGs
        y: 1D array of diagnoses (0 control, 1 positive)
        pinfo: patient info dataframe
        
    ----------
    """
    print('Loading %s Cohort Data' % cohort)
    
    X,y,pinfo = load_ECG_db(cohort,use_leads)
    X,y,pinfo = add_phenotype_info(pinfo,X,y,pinfo_csv,remove_missing_phenotype)
        
    print('Completed Loading %s Cohort ECG Data' % cohort)
    return X,y,pinfo

def load_ECG_db(ajm_bas,use_leads):
    """
    Loads ECG database directly from the SQL server
    ----------
    Inputs:
        ajm_bas: either 'Basale' or 'Ajmaline' to identify the ECG cohort to use
        
    Outputs:
        X: 2D matrix of ECGs
        y: 1D array of diagnoses (0 control, 1 positive)
        pinfo: patient info dataframe
        
    ----------
    """
    # print('Loading ECG data')
    # Connect to SQL server database
    session = mysql.connector.connect(host='localhost',                           
                                 database='brugada',
                                 user='root',
                                 password='root',
                                 use_pure=True)
    cursor = session.cursor(buffered=True)
    # loop through SQL fields of interest to extract patient info into dataframe
    if 'bas' in ajm_bas.lower():
        ajm_bas_HR = 'Baseline_HR'
    elif 'ajm' in ajm_bas.lower():
        ajm_bas_HR = 'Ajmaline_HR'
    SQL_fields = ['PatientID','Diagnosis','ECGType','HR','HRV']
    pinfo_SQL = pd.DataFrame(columns=SQL_fields)
    for field_ix,field in enumerate(SQL_fields[:-1]):
        if field == 'HR':
            sql_fetch_patients = """SELECT %s FROM `patient_info`""" % ajm_bas_HR
        else:
            sql_fetch_patients = """SELECT %s FROM `patient_info`""" % field
        cursor.execute(sql_fetch_patients)
        pinfo_SQL[field] = [ii[0] for ii in cursor.fetchall()]
            
    # loop through all patients and grab ECG data
    df_ECGs = pd.DataFrame(columns=use_leads)
    df_ECGs['PatientID'] = read_pIDs(session, cursor,3, ajm_bas)
    missing_data_pixs = []
    for pix,pID in enumerate(df_ECGs['PatientID']):
        for lead in use_leads:
            # read ECG median heart beat from SQL database
#            print( readBLOB(session, cursor, pID, 3, ajm_bas, lead))
            df_ECGs[lead][pix] = readBLOB(session, cursor, pID, 3, ajm_bas, lead)
            if len(df_ECGs[lead][pix]) == 0:
                missing_data_pixs.append(pix)
    complete_pixs = [pix for pix,pID in enumerate(df_ECGs['PatientID']) if not(pix in missing_data_pixs)]        
    df_ECGs = df_ECGs.iloc[complete_pixs].reset_index().drop(['index'],axis=1)
    
    # Concatenate ECGs together
    X = fuse_ECGs(df_ECGs,use_leads)
    
    # Filter the pinfo with patient numbers with ECG data
    pinfo_ixs = [pix for pix,pID in enumerate(pinfo_SQL['PatientID']) if pID in np.asarray(df_ECGs['PatientID'])]
    pinfo_SQL = pinfo_SQL.iloc[pinfo_ixs].reset_index().drop(['index'],axis=1)
    y = np.asarray(pinfo_SQL['Diagnosis'])
    
    # Add HRV info for each patient
    hrv = list()
    patients = np.asarray(pinfo_SQL['PatientID'])
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
            QRS_std = np.std(np.diff(QRS_peaks))
            row = [patient_id, 120000/QRS_med, QRS_std/QRS_med]
            hrv.append(row)
        except:
            continue
    hrv = [hrv_ix[2] for ix,hrv_ix in enumerate(hrv)]
    pinfo_SQL['HRV'] = hrv

    #% close the SQL connection
    cursor.close()
    session.close()
    
    # Normalize
    X = renormalize_individual_traces(X,len(use_leads))
    
    # ECG Signal Resampling
    f_new = 200
    X = ECG_resample(X,f_new)
    
    print('Completed Loading ECG Data')
    
    return X,y,pinfo_SQL

def read_pIDs(session, cursor, cut, ajbase):
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
        
    Output:
        record-- PatientIDs array
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
            ajm_bas_sql = 'baseline'
        elif "ajm" in ajbase.lower():
            ajm_bas_sql = 'ajmaline'
            
        sql_fetch_blob_query = """SELECT `%s` FROM `%s%s`""" % ('PatientID',ajm_bas_sql,cut_tbl)    
        
        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
        record = [tp[0] for tp in record]
        return record
    except mysql.connector.Error as error:
        session.rollback()
        print("Failed to read BLOB data from MySQL table {}".format(error))

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
            sql_fetch_blob_query = """SELECT `"""+lead+"""` FROM `baseline"""+cut_tbl+"""` WHERE PatientID = %d"""
        elif "ajm" in ajbase.lower():
            sql_fetch_blob_query = """SELECT `"""+lead+"""` FROM `ajmaline"""+cut_tbl+"""` WHERE PatientID = %d"""
        cursor.execute(sql_fetch_blob_query % patient)
        blob = cursor.fetchall()
        trace = []
        try:
            trace = np.frombuffer(blob[0][0]).reshape([-1,1500]).transpose() # 2, 1 grid
            trace = np.asarray([num[0] for num in trace])
        except:
            try:
                trace = [float(ii) for ii in blob[0][0].split()] # 0
                trace = np.asarray([num[0] for num in trace])
            except:
                try:
                    trace = np.frombuffer(blob[0][0]) # 1 (long)
                    trace = np.asarray([num[0] for num in trace])
                except:
                    None
        return trace
    except mysql.connector.Error as error:
        session.rollback()
        print("Failed to read BLOB data from MySQL table {}".format(error))

def fuse_ECGs(df_ECGs,use_leads):
    """
    Produces fused data vector with specified leads    
    
    Inputs:
        df_ECGs-- dataframe containing average lead beats
        use_leads-- list of ltrings to fuse together
        
    Output:
        X-- Fused ECG data
    """
#    lead_strs = ['I','II']#,'III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']
    X = np.zeros(shape=(len(df_ECGs),1500*len(use_leads)))
    for ix,lead_str in enumerate(use_leads):
        X[:,ix*1500:(ix+1)*1500] = np.stack(np.asarray(df_ECGs[lead_str]))
    return X

def renormalize_individual_traces(X,nleads):
    """
    Renormalizes leads individually to have standard deviation = 1    
    
    Inputs:
        X-- ECG data
        nleads-- total number of leads in X
        
    Output:
        X-- Normalized ECG data
    """
    X_renorm = np.zeros(shape=(X.shape[0],X.shape[1]))
    for kk in range(X.shape[0]):
        for jj in range(nleads):
            lead_trace = X[kk,jj*1500:(jj+1)*1500-1]
            if lead_trace.std() != 0:
                X_renorm[kk,jj*1500:(jj+1)*1500-1] = (lead_trace-lead_trace.mean())/lead_trace.std()
    return X_renorm

def ECG_resample(X,f_new):
    """
    Resamples X to new frequency using FFT    
    
    Inputs:
        X-- ECG data
        f_new-- new sampling frequency of X
        
    Output:
        X-- Resampled ECG data
    """
    # Resample function from 2000Hz to f_new 
    resample_length = int(X.shape[1]*f_new/2000)
    print('Resampling ECG signal from %d Hz to %d Hz' % (2000,f_new))
    X_filt = signal.resample(X, resample_length, t=None, axis=1, window=None)
    print('Resampling Complete')
    plt.subplot(211)
    plt.plot(X[0,:])
    plt.title("Original 2000Hz")
    plt.subplot(212)
    plt.plot(X_filt[0,:])
    plt.title('Resampled %dHz' % f_new)
    plt.show()
    return X_filt

def add_phenotype_info(pinfo,X,y,pinfo_csv_file,remove_missing_phenotype):
    """
    Loads ECG database directly from the SQL server
    ----------
    Inputs:
        pinfo-- patient info dataframe
        X-- 2D matrix of ECGs
        y-- 1D array of diagnoses (0 control, 1 positive)
        pinfo_csv_file-- file path for csv file containing patient info
        remove_missing_phenotype-- [Boolean] remove patients with incomplete info?
        
    Outputs:
        X: 2D matrix of ECGs
        y: 1D array of diagnoses (0 control, 1 positive)
        pinfo: patient info dataframe
        
    ----------
    """
    # Load up the XLS file with phenotype information
    pinfo_csv = pd.read_csv(pinfo_csv_file,header=0, keep_default_na=False)
    pinfo_csv_ixs = np.asarray(pinfo['PatientID']-1)
    pinfo['Phenotype'] = list(pinfo_csv['EKGType'][pinfo_csv_ixs])
    pinfo = pinfo.drop(['ECGType'],axis=1)
    # Remove patients with missing phenotype data if specified
    if remove_missing_phenotype:
        good_ixs = [ix for ix,phenotype in enumerate(pinfo['Phenotype']) if not(phenotype == '')]
        pinfo = pinfo.iloc[good_ixs].reset_index().drop(['index'],axis=1)
        X = X[good_ixs]
        y = np.asarray(pinfo['Diagnosis'])
    return X,y,pinfo

def row_shuffle(X,y):
    """
    Randomy shuffles the X and y matrices, maintains index correlation
    ----------
    Inputs:
        X-- 2D matrix of ECGs
        y-- 1D array of diagnoses (0 control, 1 positive)
        
    Outputs:
        X: 2D matrix of ECGs shuffled
        y: 1D array of diagnoses (0 control, 1 positive) shuffled
        
    ----------
    """
    # Permute X and y rows to randomize sample order
    nrows = X.shape[0]
    X_shuffle = np.zeros(shape=(nrows,X.shape[1]))
    y_shuffle = np.zeros(shape=(nrows,))
    shuffle_indxs = [x for x in range(nrows)]
    np.random.shuffle(shuffle_indxs)
    for ii,new_indx in enumerate(shuffle_indxs):
        X_shuffle[ii,:] = X[new_indx,:]
        y_shuffle[ii] = y[new_indx]
    return X_shuffle,y_shuffle

#%% DNN Training Functions

def LOOCV_TRAIN_FN(holdout_index,save_models,save_path,pinfo,X,y):
    """
    Main DNN training function for LOOCV (leave one out cross validation)
    ----------
    Inputs:
        holdout_index-- [Int] patient index to use as holdout for LOOCV
        save_models-- [Boolean] save h5 and json files to save_path?
        save_path-- directory where results of the DNN are stored
        pinfo: patient info dataframe
        X-- 2D matrix of ECGs
        y-- 1D array of diagnoses (0 control, 1 positive)
        
    Outputs:
        holdout_reg-- array of DNN output scores for the holdout patient
        training_stats_reg-- 2D array of DNN output scores for training patients
        training_stats-- 2D array of number of correct diagnoses for training patients
        holdout_stats-- array of number of correct diagnoses for holdout patient
        
    ----------
    """
    # NN sturcture info
    batch_size = 16 # batch size
    layers = 3 # number of layers
    nodes = 5 # number of nodes per layer
    optimizer = 'adagrad' # gradient optimizer
    
    # plotting during training
    plot_training = False
    
    # Early stopping when there are diminishing returns in accuracy with epoch
    early_stop = True
    
    # Shuffle y matrix to test for overfitting
    y_scramble = False
    
    # Select cross validation method for training
    cv_method = 'kfold'
    n_folds = 7
    
    # Apply isotonic calibration to DNN output for basale cohort
    isotonic_cal = 'bas' in save_path.lower()
    
    # Processing patient information    
    print('Processing Patient %d of %d: %d' % \
          (holdout_index+1,X.shape[0],pinfo['PatientID'][holdout_index]))
    # process LOOCV with holdout index    
    X_nn, y_nn, X_holdout, y_holdout, nn_index = split_holdout_one_by_one(X,y,holdout_index)                    
    # Fit with cross-validation
    if cv_method == 'kfold':
        # prepare the stratified k-fold cross-validation configuration
        kfold = StratifiedKFold(n_folds, False, None)
        # cross validation estimation of performance
        scores, members, histories, irs = list(), list(), list(), list()
        for train_ix, test_ix in kfold.split(X_nn,y_nn):
        	# select samples
            trainX, trainy = X_nn[train_ix], y_nn[train_ix]
            testX, testy = X_nn[test_ix], y_nn[test_ix]
            # evaluate model
            if y_scramble:
                trainy = np.random.permutation(trainy)
                testy = np.random.permutation(testy)
            # Fit the model
            model, history, test_acc, ir = fit_model_v2(trainX, trainy, testX, testy, batch_size, layers, nodes, optimizer, early_stop,isotonic_cal)
                
            if plot_training:
                plt.figure(figsize=(10,6))
                # plot loss during training
                plt.subplot(211)
                plt.title('ALL PATIENT: batch= ' + str(batch_size) + ' | layers= '+ str(layers) + ' | nodes= ' + str(nodes) + ' | L1 = 1E-2 | opt= '+ optimizer) 
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='test')
                plt.legend()
                # plot accuracy during training
                plt.subplot(212)
                plt.title('Accuracy')
                plt.plot(history.history['accuracy'], label='train')
                plt.plot(history.history['val_accuracy'], label='test')
                plt.legend()
                plt.show()
            
            print('>%.3f' % test_acc)
            scores.append(test_acc)
            members.append(model)
            histories.append(history)
            if isotonic_cal:
                irs.append(ir)
  
    # summarize expected performance
    print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    

    holdout_reg = np.zeros(shape=(X.shape[0],n_folds))
    # Evaluate models on holdout patient
    if isotonic_cal:
        holdout_reg[holdout_index,:] = np.asarray([irs[kk].predict([model.predict(np.expand_dims(X_holdout,axis=0))[0][0]]) for kk ,model in enumerate(members)]).flatten()
    else:
        holdout_reg[holdout_index,:] = np.asarray([model.predict(np.expand_dims(X_holdout,axis=0))[0][0] for kk ,model in enumerate(members)]) 
    
    print(holdout_reg[holdout_index,:])
    
    training_stats_reg = np.zeros(shape=(X.shape[0],n_folds))
    training_stats = np.zeros(shape=(X.shape[0],4)) 
    holdout_stats = np.zeros(shape=(X.shape[0],4)) 
    iter_ix = 0               
    for model_ix,model in enumerate(members):
        if save_models:
            model.save_weights(os.path.join(save_path,'model_' + str(n_folds*holdout_index+iter_ix) + '.h5'))
            model_json = model.to_json()
            with open(os.path.join(save_path,'model_' + str(n_folds*holdout_index+iter_ix) + '.json'), "w") as json_file:
                json_file.write(model_json)
            json_file.close()
            
        if isotonic_cal:
            y_pred = np.round(irs[model_ix].predict(model.predict(X).flatten())).flatten()
            if save_models:
                f = open(os.path.join(save_path,'ir_cal_' + str(n_folds*holdout_index+iter_ix) + '.txt'), "wb")
                pickle.dump(irs[model_ix],f)  # pickle
                f.close()
        else:
            y_pred = np.round(model.predict(X)).flatten()    

        for jj in range(X.shape[0]):
            if y_pred[jj] == 1.0:
                if y[jj] == 1.0:
                    training_stats[jj,1] += 1   # true positive
                else:
                    training_stats[jj,3] += 1   # false positive
                #holdouts    
                if y[jj] == 1.0 and jj == holdout_index:
                    holdout_stats[jj,1] += 1   # true positive
                elif y[jj] == 0.0 and jj == holdout_index:
                    holdout_stats[jj,3] += 1   # false positive    

            elif y_pred[jj] == 0.0:
                if y[jj] == 1.0:
                    training_stats[jj,2] += 1   # false negative
                else:
                    training_stats[jj,0] += 1   # true negative
                #holdout
                if y[jj] == 1.0 and jj == holdout_index:
                    holdout_stats[jj,2] += 1   # false negative
                elif y[jj] == 0.0 and jj == holdout_index:
                    holdout_stats[jj,0] += 1   # true negative
        if isotonic_cal:
            training_stats_reg[:,iter_ix] = irs[model_ix].predict(model.predict(X).flatten()).flatten()
        else: 
            training_stats_reg[:,iter_ix] = model.predict(X).flatten()
        # training_stats_reg[:,iter_ix] = model.predict(X).flatten()
        iter_ix+=1
  
    K.clear_session()
    return [holdout_reg, training_stats_reg, training_stats, holdout_stats]

def split_holdout_one_by_one(X,y,holdout_index):
    """
    Main DNN training function for LOOCV (leave one out cross validation)
    ----------
    Inputs:
        X-- 2D matrix of ECGs
        y-- 1D array of diagnoses (0 control, 1 positive)
        holdout_index-- [Int] patient index to use as holdout for LOOCV
        
    Outputs:
        X_nn-- 2D matrix of training ECG data
        y_nn-- 1D training target diagnosis vector
        X_holdout-- ECG data for holdout patient
        y_holdout-- diagnosis of holdout patient
        nn_index-- matrix indicies for all patients except holdout patient
        
    ----------
    """
    # Generate indices for matrix segmentation and then segment
    nn_index = [x for x in range(X.shape[0]) if x != holdout_index]
    X_nn, X_holdout = X[nn_index], X[holdout_index]
    y_nn, y_holdout = y[nn_index], y[holdout_index] 
    
    return X_nn, y_nn, X_holdout, y_holdout, nn_index

def fit_model_v2(trainX, trainy, testX, testy, batch_size, layers, nodes, optimizer, early_stop, isotonic_cal):
    """""
    Alternative of fit_model where the output DNN score is calibrated
    
    """""
    plot = False
    # create NN model for brugada
    model = Sequential()
    model.name = 'Brugada ECG NN'
    
    for kk in range(layers):
        if kk == 0:
            model.add(Dense(nodes, input_dim=trainX.shape[1], kernel_regularizer=l1(1E-2)))  
        else:
             model.add(Dense(nodes, kernel_regularizer=l1(1E-2)))
        model.add(GaussianNoise(0.1))
        model.add(Activation('relu'))   
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # train model with option of early stopping
    if early_stop:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=10000, batch_size=batch_size, verbose=0, callbacks=[es])
    else:
        history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, batch_size=batch_size, verbose=0)
        
    # calibration of output
    probs = model.predict(trainX).flatten()
    fop, mpv = calibration_curve(trainy, probs, n_bins=10, normalize=True)
    if plot:
        # plot perfectly calibrated
        plt.figure(figsize=(10,6))
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot model reliability
        plt.plot(mpv, fop, marker='.')
        # plt.show()
    ir = IsotonicRegression()
    if isotonic_cal:
        ir.fit(probs,trainy)
        # plot model reliability
        if plot:
            results_ir = ir.predict(probs)
            fop1, mpv1 = calibration_curve(trainy, results_ir, n_bins=10, normalize=True)
            plt.plot(mpv1, fop1, marker='.')
            plt.ylabel('Fraction of Positives')
            plt.xlabel('Mean Predicted Value')
            plt.legend(['Perfect Calibration','Uncalibrated DNN','Isotonic Calibration'])
            plt.title('Calibration Plots (reliability curve)')
            plt.show()

    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('NN Training Complete')
    return model, history, test_acc, ir


