"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Brugada SQL input script
%
% Ashton Christy
# Luke Melo
% 18 May 2019
% 
% Version 4.0
% Edited 14 Feb 2021
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Load packages
import os, time, mysql.connector, pandas as pd, ecglib as ecg

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
                             password='root')
cursor = session.cursor()

# Build SQL tables
ecg.sql_reset(session, cursor)

#%%
# Load data from folder
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
      "%                                                      %\n" + \
      "%      Please use dialog window to select folder       %\n" + \
      "%               containing patient data.               %\n" + \
      "%                                                      %\n" + \
      "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
try:
    datadir = ecg.get_path("Z:\\Brugada_Project\\All_EKG_Data\\ECG_text_files") + "\\"
except:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
          "%                                                      %\n" + \
          "%                 No folder selected!                  %\n" + \
          "%                                                      %\n" + \
          "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    input("Press Enter to exit...")
    exit(0)

#%%

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
      "%                                                      %\n" + \
      "%       Please use dialog window to select .XLSX       %\n" + \
      "%             file containing patient data.            %\n" + \
      "%                                                      %\n" + \
      "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
try:
    xlfile = ecg.get_file("Excel File (*.xlsx)|*.xlsx", "Z:\\Brugada_Project\\All_EKG_Data\\")
    xldb = pd.read_excel(xlfile)
except:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
          "%                                                      %\n" + \
          "%               Invalid file selected!                 %\n" + \
          "%                                                      %\n" + \
          "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    input("Press Enter to exit...")
    exit(0)
    
#%%
 
row_list = []  
for row in xldb.itertuples():
    row = list(row)
    if str(row[1]).isdigit():
        row_list.append(row[0])
    
# Append patient info from CSV to database
for rows in xldb.itertuples():
    if rows[0] in row_list:
        row = list(rows)
        patientID = row[1]
        info_query = """ INSERT INTO `patient_info` (`PatientID`, `Diagnosis`)
                        VALUES (%s,%s); """
        if row[-1] != row[-1]: # isnan
            row[-1] = None
        try:
            cursor.execute(info_query, ([ii for ii in [patientID,row[4]]]))
            session.commit()
        except mysql.connector.Error as error:
            if row[2] == False:
                try:
                    cursor.execute(info_query, ([ii for ii in [patientID,row[4]]]))
                    session.commit()
                except:
                    session.rollback()
                    print("Patient info for "+str(patientID)+" not inserted: {}".format(error))
                    print([ii for ii in [patientID,row[4]]])
            else:
                session.rollback()
                print("Patient info for "+str(patientID)+" not inserted: {}".format(error))
                print([ii for ii in [patientID,row[4]]])

#%%

# Initialize variables
patientlist = []
patientpheno = []
sql_row_tab = []
datasets = ["bas","ajm"]

# Define leads of interest
leads = ["I","II","III","V1","V2","V3","V4","V5","V6","aVF","aVL","aVR"]

# Read through folders
folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
folderlist.sort()

# Loop through folders
for ii,folder in enumerate(folderlist):
    
    # Exclude base dir
    try:
        folder.split('\\')[-1][0]
    except:
        continue

    # Initialize SQL query for each patient - verify folder has bas/ajm AND patient number
    if folder.split('\\')[-2].isdigit() and \
        any([xx in folder.lower() for xx in datasets]):    

        files = [jj for jj in os.listdir(folder)]
        files.sort()
        
        # Get patient number
        sql_row_init = [folder.split('\\')[-2]]
        patientlist.append(folder.split('\\')[-2])
        xlrow = xldb[xldb['Patient Number'] == int(folder.split('\\')[-2])]
        if not len(xlrow):
            print("Patient data for "+str(sql_row_init[0])+" not loaded: patient missing from Excel file!")
            continue

        # Get diagnosis from file
        sql_row_init += str(xlrow.iloc[0]['Diagnosis'])
        sql_row_tab.append(sql_row_init)

        print("Loading patient "+patientlist[-1]+" ("+folder.split('\\')[-1][0:3].lower()+")")

        # Segregate and update SQL
        if "bas" in folder.split('\\')[-1].lower():
            init_query = """INSERT INTO `baseline_traces` (`PatientID`, `Diagnosis`)
                            VALUES (%s,%s);"""
            try:
                cursor.execute(init_query, ([xx for xx in sql_row_init]))  
                session.commit()
            except mysql.connector.Error as error:
                session.rollback()
                print("Patient info for "+str(sql_row_init[0])+" not inserted: {}".format(error))
        elif "ajm" in folder.split('\\')[-1].lower():
            init_query = """INSERT INTO `ajmaline_traces` (`PatientID`, `Diagnosis`)
                            VALUES (%s,%s);"""
            try:
                cursor.execute(init_query, ([xx for xx in sql_row_init]))  
                session.commit()
            except mysql.connector.Error as error:
                session.rollback()
                print("Patient info for "+str(sql_row_init[0])+" not inserted: {}".format(error))                

        # Build SQL query and append to database
        for kk,file in enumerate(files):
            if "info" not in file.lower():
                page = file.split("Page ")[1].split(".")[0]
                ecg_session = file.split(".")[1].split("Session ")[1].split(" ")[0]
                for lead in leads:
                    if "bas" in folder.split('\\')[-1].lower():
                        update_query = """ UPDATE `baseline_traces`
                                        SET `Session` = (%s), `Page` = (%s)
                                        WHERE `PatientID` = %s; """   
                        try:
                            cursor.execute(update_query, (ecg_session, page, str(folder.split('\\')[-2]), ))
                            session.commit()
                        except mysql.connector.Error as error:
                            session.rollback()
                            print("ECG info for "+str(folder.split('\\')[-2])+" not inserted: {}".format(error))                            
                    elif "ajm" in folder.split('\\')[-1].lower():
                        update_query = """ UPDATE `ajmaline_traces`
                                        SET `Session` = (%s), `Page` = (%s)
                                        WHERE `PatientID` = %s;"""  
                        try:
                            cursor.execute(update_query, (ecg_session, page, str(folder.split('\\')[-2]), ))
                            session.commit()
                        except mysql.connector.Error as error:
                            session.rollback()
                            print("ECG info for "+str(folder.split('\\')[-2])+" not inserted: {}".format(error))
                    
                    if lead in file:
                        ecg.insertBLOB(session, cursor, folder.split('\\')[-2], \
                               folder.split('\\')[-1], lead, folder+'\\'+file)       

#%%

# Copy info between tables
ecg.sql_copy(session, cursor)

#%%

cursor.close()
session.close()

# Folder name split formula:                 
# -3 = control/patient
# -2 = patient
# -1 = ajm/base
t1 = time.time()
total = t1-t0
print("\nDone.")
print("Total time elapsed :", total, "\n")
##%%
input("Press Enter to exit...")
exit(0)