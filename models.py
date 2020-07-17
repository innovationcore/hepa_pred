import json
import uuid
from statistics import stdev
import keras
import pickle
from sklearn.model_selection import KFold, cross_validate, RepeatedKFold
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree, metrics
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz

def getsample(file_path):

    dataset = pd.read_csv(file_path)

    # dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    # print(dataset.columns[dataset.isna().any()].tolist())

    # print("pre NA drop: " + str(dataset.shape))
    # dataset = dataset.dropna()
    # print("post NA drop: " + str(dataset.shape))

    # dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    # dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    '''
y.Admission.INR
y.Adm.Cr
x.Cirrhosis
y.Adm.WBC
y.Adm.ALT
    '''

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    # X = X.drop(['y.BMI'], axis=1)

    drop_list = ['x.HEP.C.RNA_N', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.Na',
                 'x.Cirrhosis_N', 'x.DM_N', 'z.bad_N']

    # 'x.Cirrhosis_N', 'x.Cirrhosis_Y', 'y.Adm.Na', 'y.Adm.ALT', 'y.Adm.ALP', 'y.Adm.Alb',
    for term in drop_list:
        dataset = dataset.drop([term], axis=1)

    pd.set_option('display.max_columns', None)

    print(dataset)
    t = 0
    f = 0
    for yy in y:
        if yy == 1:
            t += 1
        else:
            f += 1

    print("t=" + str(t) + " " + " f=" + str(f))


def testmodel(file_path):

    dataset = pd.read_csv(file_path)

    # dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    # print(dataset.columns[dataset.isna().any()].tolist())

    # print("pre NA drop: " + str(dataset.shape))
    # dataset = dataset.dropna()
    # print("post NA drop: " + str(dataset.shape))

    # dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    # dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    '''
y.Admission.INR
y.Adm.Cr
x.Cirrhosis
y.Adm.WBC
y.Adm.ALT
    '''

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    #X = X.drop(['z.bad_N'], axis=1)
    # X = X.drop(['y.BMI'], axis=1)

    drop_list = ['x.HEP.C.RNA_N', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.Na',
                 'x.Cirrhosis_N', 'x.DM_N', 'z.bad_N']

    # 'x.Cirrhosis_N', 'x.Cirrhosis_Y', 'y.Adm.Na', 'y.Adm.ALT', 'y.Adm.ALP', 'y.Adm.Alb',
    for term in drop_list:
        X = X.drop([term], axis=1)

    pd.set_option('display.max_columns', None)

    RF_model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    #for final testing
    Xs = scaler.fit_transform(X)
    ys = y

    scores = cross_validate(RF_model, Xs, ys, scoring=('precision', 'recall', 'roc_auc', 'accuracy'))

    cv_sensitivity = np.mean(scores['test_recall'])
    cv_specificity = np.mean(scores['test_precision'])
    cv_auc = np.mean(scores['test_roc_auc'])
    cv_acc = np.mean(scores['test_accuracy'])

    t_sensitivity, t_specificity, t_auc, t_acc, t_ap, t_kappa = getmodelstats(RF_model, Xs, ys)

    print("cv_sensitivity: " + str(cv_sensitivity))
    print("t_sensitivity: " + str(t_sensitivity))
    print("--")
    print("cv_specificity: " + str(cv_specificity))
    print("t_specificity: " + str(t_specificity))
    print("--")
    print("cv_auc: " + str(cv_auc))
    print("t_auc: " + str(t_auc))
    print("--")
    print("cv_accuracy: " + str(cv_acc))
    print("t_accuracy: " + str(t_acc))


def getmodelstats(model, X, y):

    test_pred = model.predict(X)

    acc = metrics.accuracy_score(y, test_pred)
    kappa = metrics.cohen_kappa_score(y, test_pred)
    ap = metrics.average_precision_score(y, test_pred)
    auc = metrics.roc_auc_score(y, test_pred)

    tn, fp, fn, tp = confusion_matrix(y, test_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return sensitivity, specificity, auc, acc, ap, kappa

def getrandomforest(file_path):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    print("pre NA drop: " + str(dataset.shape))
    dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))


    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns
    print(dataset.iloc[:, 5:].head(5))

    droplist = ['z.bad_Y', 'y.BMI']
    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)

    savecol = dataset[['z.bad_Y',]].copy()

    # Split the dataset to trainand test data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

    parameters = {'bootstrap': True,
                  'min_samples_leaf': 3,
                  'n_estimators': 50,
                  'min_samples_split': 10,
                  'max_features': 'sqrt',
                  'max_depth': 6,
                  'max_leaf_nodes': None}

    RF_model = RandomForestClassifier(**parameters)
    RF_model.fit(train_X, train_y)
    test_pred = RF_model.predict(test_X)

    acc = metrics.accuracy_score(test_y, test_pred)
    kappa = metrics.cohen_kappa_score(test_y, test_pred)
    ap = metrics.average_precision_score(test_y, test_pred)
    auc = metrics.roc_auc_score(test_y, test_pred)

    '''
    print(test_y)
    print(classification_report(test_y, test_pred, labels=[0, 1]))
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(sensitivity)
    print(specificity)
    '''

    tree_len = len(RF_model.estimators_)

    estimator = RF_model.estimators_[tree_len -1]

    print(X.columns)
    print(RF_model.classes_)

    # Export as dot file
    dotfile = export_graphviz(estimator,
                    feature_names=X.columns,
                    class_names = True,
                    filled=True,
                    rounded=True)

    text_file = open("tree.dot", "w")
    text_file.write(dotfile)
    text_file.close()

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'decision_tree.png', '-Gdpi=600'])

    '''
    fn = X.columns
    cn = RF_model.classes_
    cn = True
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(RF_model.estimators_[0],
                   feature_names=fn,
                   class_names=cn,
                   filled=True);
    fig.savefig('rf_individualtree.png')


    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'decision_tree_89.png', '-Gdpi=600'])
    '''

    return acc, ap, auc, kappa

def getrandomforeststats_fold_cv(file_path, loops):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    #print(dataset.columns[dataset.isna().any()].tolist())

    #print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    #print("post NA drop: " + str(dataset.shape))

    #dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    #dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    #X = X.drop(['y.BMI'], axis=1)

    '''
y.Admission.INR
y.Adm.Cr
x.Cirrhosis
y.Adm.WBC
y.Adm.ALT
    '''

    # drop_list = ['x.HEP.C.RNA_N','y.Adm.WBC','x.GEND_F','x.GEND_M','x.HEP.C.RNA_P','x.Hep.B_N','x.Hep.B_P','y.Adm.Platelet','y.Adm.Alb','x.alcohol.use_HEAV<6M','x.alcohol.use_HEAV>6M','y.Adm.AST','y.Adm.ALP','x.alcohol.use_MOD','x.alcohol.use_SOC','y.AGE','y.BMI','y.Adm.Bili','y.Admission.INR','x.DM_Y','x.HTN_N','x.HTN_Y','y.Adm.Cr','y.Adm.Na','y.Adm.ALT','x.Cirrhosis_N','x.Cirrhosis_Y','x.DM_N']

    drop_list = ['x.HEP.C.RNA_N', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.Na',
                 'x.Cirrhosis_N','x.DM_N']


    #'x.Cirrhosis_N', 'x.Cirrhosis_Y', 'y.Adm.Na', 'y.Adm.ALT', 'y.Adm.ALP', 'y.Adm.Alb',
    for term in drop_list:
        X = X.drop([term], axis=1)


    pd.set_option('display.max_columns', None)

    print(X.head())
    #scale values
    scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)

    #for final testing
    #Xs = scaler.fit_transform(X)
    #ys = y

    #write reults file
    f = open("models.csv", "w")
    f.write("model_id,sensitivity_all,sensitivity_model,specificity_all,specificity_model,auc_all,auc_model\n")

    #kf = KFold(n_splits=kfolds)
    #kf = RepeatedKFold(n_splits=kfolds, n_repeats=5000)
    #kf.get_n_splits(X)

    high_s = 0

    for x in range(loops):

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        Xs = scaler.transform(X)
        ys = y

        '''
        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 50,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 6,
                      'max_leaf_nodes': None}

        '''
        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 25,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 10,
                      'max_leaf_nodes': None}


        RF_model = RandomForestClassifier(**parameters)
        RF_model.fit(train_X, train_y)

        scores = cross_validate(RF_model, Xs, y, scoring=('precision', 'recall', 'roc_auc', 'accuracy'))

        cv_sensitivity = np.mean(scores['test_recall'])
        cv_specificity = np.mean(scores['test_precision'])
        cv_auc = np.mean(scores['test_roc_auc'])
        cv_acc = np.mean(scores['test_accuracy'])

        sensitivity_cut = 0.78
        specificity_cut = 0.85
        auc_cut = 0.85

        if ((cv_sensitivity > sensitivity_cut) and (cv_specificity > specificity_cut) and (cv_auc > auc_cut)):

            if high_s < cv_sensitivity:
                high_s = cv_sensitivity

                id = str(uuid.uuid1())
                with open("model_" + id + ".pkl", 'wb') as mf:
                    pickle.dump(RF_model, mf)
                with open("scale_" + id + ".pkl", 'wb') as sf:
                    pickle.dump(scaler, sf)
                #writeline = id + "," + str(sensitivity) + "," + str(m_sensitivity) + "," + str(specificity) + "," + str(m_specificity) + ","  + str(auc) + ","  + str(m_auc) + "\n"
                #f.write(writeline)

                t_sensitivity, t_specificity, t_auc, t_acc, t_ap, t_kappa = getmodelstats(RF_model, Xs, ys)
                m_sensitivity, m_specificity, m_auc, m_acc, m_ap, m_kappa = getmodelstats(RF_model, test_X, test_y)

                print("")
                print("id: " + id)
                print("cv_sensitivity: " + str(cv_sensitivity))
                print("t_sensitivity: " + str(t_sensitivity))
                print("m_sensitivity: " + str(m_sensitivity))
                print("--")
                print("cv_specificity: " + str(cv_specificity))
                print("t_specificity: " + str(t_specificity))
                print("m_specificity: " + str(m_specificity))
                print("--")
                print("cv_auc: " + str(cv_auc))
                print("t_auc: " + str(t_auc))
                print("m_auc: " + str(m_auc))
                print("--")
                print("cv_accuracy: " + str(cv_acc))
                print("t_accuracy: " + str(t_acc))
                print("m_accuracy: " + str(m_acc))

            '''
                for train_index, test_index in kf.split(X):
                
                    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
                    scaler.fit(train_X)
                    train_X = scaler.transform(train_X)
                    test_X = scaler.transform(test_X)

                    train_y, test_y = y.iloc[train_index], y.iloc[test_index]


                    parameters = {'bootstrap': True,
                                  'min_samples_leaf': 3,
                                  'n_estimators': 50,
                                  'min_samples_split': 10,
                                  'max_features': 'sqrt',
                                  'max_depth': 6,
                                  'max_leaf_nodes': None}
                    '''

            '''
            m_test_pred = RF_model.predict(test_X)

            m_acc = metrics.accuracy_score(test_y, m_test_pred)
            m_kappa = metrics.cohen_kappa_score(test_y, m_test_pred)
            m_ap = metrics.average_precision_score(test_y, m_test_pred)
            m_auc = metrics.roc_auc_score(test_y, m_test_pred)

            tn, fp, fn, tp = confusion_matrix(test_y, m_test_pred).ravel()
            m_specificity = tn / (tn + fp)
            m_sensitivity = tp / (tp + fn)
            if ((m_sensitivity > sensitivity_cut) and (m_specificity > specificity_cut) and (m_auc > auc_cut)):
                id = str(uuid.uuid1())
                with open("model_" + id + ".pkl", 'wb') as mf:
                    pickle.dump(RF_model, mf)
                with open("scale_" + id + ".pkl", 'wb') as sf:
                    pickle.dump(scaler, sf)
                writeline = id + "," + str(sensitivity) + "," + str(m_sensitivity) + "," + str(specificity) + "," + str(m_specificity) + ","  + str(auc) + ","  + str(m_auc) + "\n"
                f.write(writeline)
                print("All: " + str(sensitivity) + "," + str(specificity) + "," + str(auc))
                print("Model: " + str(m_sensitivity) + "," + str(m_specificity) + "," + str(m_auc))
                print("m_acc: " + str(m_acc) + " m_kappa: " + str(m_kappa) + " m_ap: " + str(m_ap))
                #print("acc: " + str(acc) + " kappa: " + str(kappa) + " ap: " + str(ap))
                print("acc: " + str(acc))
                print("\n")
                scores = cross_validate(RF_model, X, y, scoring = ('precision', 'recall','roc_auc','accuracy'))
                print(scores)
                #print(np.mean(scores['test_fit']))
                print("cv_precision: " + str(np.mean(scores['test_precision'])))
                print("cv_recall: " + str(np.mean(scores['test_recall'])))
                print("cv_auc: " + str(np.mean(scores['test_roc_auc'])))
                print("cv_accuract: " + str(np.mean(scores['test_accuracy'])))
                # %%
                print(classification_report(test_y, m_test_pred, target_names=['Yes', 'No']))
                '''

    f.close()
    #print(sorted(auc_list, reverse=True))
    #return np.mean(acc_list), np.mean(kappa_list), np.mean(ap_list), np.mean(auc_list), np.mean(specificity_list), np.mean(sensitivity_list)

def getrandomforeststats_fold(file_path, kfolds):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    #print(dataset.columns[dataset.isna().any()].tolist())

    #print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    #print("post NA drop: " + str(dataset.shape))

    #dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    #dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    X = X.drop(['y.BMI'], axis=1)

    '''
    x.Cirrhosis
y.Adm.WBC
y.Admission.INR
y.Adm.ALT
y.Adm.Cr

    '''

    # drop_list = ['x.HEP.C.RNA_N','y.Adm.WBC','x.GEND_F','x.GEND_M','x.HEP.C.RNA_P','x.Hep.B_N','x.Hep.B_P','y.Adm.Platelet','y.Adm.Alb','x.alcohol.use_HEAV<6M','x.alcohol.use_HEAV>6M','y.Adm.AST','y.Adm.ALP','x.alcohol.use_MOD','x.alcohol.use_SOC','y.AGE','y.BMI','y.Adm.Bili','y.Admission.INR','x.DM_Y','x.HTN_N','x.HTN_Y','y.Adm.Cr','y.Adm.Na','y.Adm.ALT','x.Cirrhosis_N','x.Cirrhosis_Y','x.DM_N']
    drop_list = ['x.HEP.C.RNA_N', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.Na',
                'x.DM_N']

    #for term in drop_list:
    #    X = X.drop([term], axis=1)

    #scale values
    scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)

    acc_list = []
    kappa_list = []
    ap_list = []
    auc_list = []
    specificity_list = []
    sensitivity_list = []

    kf = KFold(n_splits=kfolds)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.fit_transform(test_X)

        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]

        #train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 50,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 6,
                      'max_leaf_nodes': None}

        RF_model = RandomForestClassifier(**parameters)
        RF_model.fit(train_X, train_y)
        test_pred = RF_model.predict(test_X)

        acc = metrics.accuracy_score(test_y, test_pred)
        acc_list.append(acc)
        kappa = metrics.cohen_kappa_score(test_y, test_pred)
        kappa_list.append(kappa)
        ap = metrics.average_precision_score(test_y, test_pred)
        ap_list.append(ap)
        auc = metrics.roc_auc_score(test_y, test_pred)
        auc_list.append(auc)

        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)
        sensitivity = tp / (tp + fn)
        sensitivity_list.append(sensitivity)

    #print(stdev(acc_list))
    #print(stdev(auc_list))
    return np.mean(acc_list), np.mean(kappa_list), np.mean(ap_list), np.mean(auc_list), np.mean(specificity_list), np.mean(sensitivity_list)

def getrandomforeststats_all(file_path, loopcount):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    print(dataset.columns[dataset.isna().any()].tolist())

    print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))

    dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    X = X.drop(['y.BMI'], axis=1)

    acc_list = []
    kappa_list = []
    ap_list = []
    auc_list = []
    specificity_list = []
    sensitivity_list = []

    for i in range(loopcount):
        # Split the dataset to trainand test data
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 50,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 6,
                      'max_leaf_nodes': None}

        RF_model = RandomForestClassifier(**parameters)
        RF_model.fit(train_X, train_y)
        test_pred = RF_model.predict(test_X)

        acc = metrics.accuracy_score(test_y, test_pred)
        acc_list.append(acc)
        kappa = metrics.cohen_kappa_score(test_y, test_pred)
        kappa_list.append(kappa)
        ap = metrics.average_precision_score(test_y, test_pred)
        ap_list.append(ap)
        auc = metrics.roc_auc_score(test_y, test_pred)
        auc_list.append(auc)

        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)
        sensitivity = tp / (tp + fn)
        sensitivity_list.append(sensitivity)

    #print(stdev(acc_list))
    #print(stdev(auc_list))
    return np.mean(acc_list), np.mean(kappa_list), np.mean(ap_list), np.mean(auc_list), np.mean(specificity_list), np.mean(sensitivity_list)

def getrandomforeststats_3(file_path, loopcount):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    print(dataset.columns[dataset.isna().any()].tolist())

    #print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    #print("post NA drop: " + str(dataset.shape))

    dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)

    #drop_list = ['x.HEP.C.RNA_N','y.Adm.WBC','x.GEND_F','x.GEND_M','x.HEP.C.RNA_P','x.Hep.B_N','x.Hep.B_P','y.Adm.Platelet','y.Adm.Alb','x.alcohol.use_HEAV<6M','x.alcohol.use_HEAV>6M','y.Adm.AST','y.Adm.ALP','x.alcohol.use_MOD','x.alcohol.use_SOC','y.AGE','y.BMI','y.Adm.Bili','y.Admission.INR','x.DM_Y','x.HTN_N','x.HTN_Y','y.Adm.Cr','y.Adm.Na','y.Adm.ALT','x.Cirrhosis_N','x.Cirrhosis_Y','x.DM_N']
    drop_list = ['x.HEP.C.RNA_N', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.ALT',
                 'x.Cirrhosis_N', 'x.Cirrhosis_Y', 'x.DM_N']

    #print(str(len(drop_list)))

    for term in drop_list:
        X = X.drop([term], axis=1)

    #X = X.drop(['y.BMI'], axis=1)

    #print(X)
    #exit(0)

    acc_list = []
    kappa_list = []
    ap_list = []
    auc_list = []
    specificity_list = []
    sensitivity_list = []

    for i in range(loopcount):
        # Split the dataset to trainand test data
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 50,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 6,
                      'max_leaf_nodes': None}

        RF_model = RandomForestClassifier(**parameters)
        RF_model.fit(train_X, train_y)
        test_pred = RF_model.predict(test_X)

        acc = metrics.accuracy_score(test_y, test_pred)
        acc_list.append(acc)
        kappa = metrics.cohen_kappa_score(test_y, test_pred)
        kappa_list.append(kappa)
        ap = metrics.average_precision_score(test_y, test_pred)
        ap_list.append(ap)
        auc = metrics.roc_auc_score(test_y, test_pred)
        auc_list.append(auc)

        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)
        sensitivity = tp / (tp + fn)
        sensitivity_list.append(sensitivity)

    #print(stdev(acc_list))
    #print(stdev(auc_list))
    return np.mean(acc_list), np.mean(kappa_list), np.mean(ap_list), np.mean(auc_list), np.mean(specificity_list), np.mean(sensitivity_list)

def getrandomforestave(file_path, loopcount):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    print(dataset.columns[dataset.isna().any()].tolist())

    print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))

    dataset['y.Adm.Cr'] = np.log(dataset['y.Adm.Cr'])
    dataset['y.Admission.INR'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    #X = X.drop(['y.BMI'], axis=1)


    combined_score = 0

    max = 0
    min = 1
    treedot = ""

    for i in range(loopcount):
        # Split the dataset to trainand test data
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 50,
                      'min_samples_split': 10,
                      'max_features': 'sqrt',
                      'max_depth': 6,
                      'max_leaf_nodes': None}

        RF_model = RandomForestClassifier(**parameters)
        RF_model.fit(train_X, train_y)
        RF_predictions = RF_model.predict(test_X)
        accuracy = accuracy_score(test_y, RF_predictions)
        if accuracy > max:
            max = accuracy
            #get tree.dot

            tree_len = len(RF_model.estimators_)
            estimator = RF_model.estimators_[tree_len - 1]
            dotfile = export_graphviz(estimator,
                                      feature_names=X.columns,
                                      class_names=True,
                                      filled=True,
                                      rounded=True)

        if accuracy < min:
            min = accuracy
        combined_score =  combined_score + accuracy
        #print(accuracy)


    text_file = open("tree.dot", "w")
    text_file.write(dotfile)
    text_file.close()

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'decision_tree.png', '-Gdpi=600'])

    return combined_score/loopcount, max, min

def getnn(file_path):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)
    print("pre NA drop: " + str(dataset.shape))
    dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns
    print(dataset.iloc[:, 5:].head(5))

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)

    # Split the dataset to trainand test data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

    # Build a neural network :
    NN_model = Sequential()
    NN_model.add(Dense(128, input_dim=28, activation='relu'))
    NN_model.add(Dense(256, activation='relu'))
    NN_model.add(Dense(256, activation='relu'))
    NN_model.add(Dense(256, activation='relu'))
    NN_model.add(Dense(1, activation='sigmoid'))
    NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #checkpoint_name = 'Weights-{epoch:03d}-{val_accuracy:.5f}.hdf5'
    checkpoint_name = 'checkpoint.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    NN_model.fit(train_X, train_y, epochs=40, batch_size=64, validation_split=0.2, callbacks=callbacks_list)


    #wights_file = './Weights-016-0.88060.hdf5'  # choose the best checkpoint
    NN_model.load_weights(checkpoint_name)  # load it
    NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    predictions = NN_model.predict(test_X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    predictions = rounded
    score = accuracy_score(test_y, predictions)
    return(score)

def getnnstats(file_path, kfolds):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)
    print("pre NA drop: " + str(dataset.shape))
    dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))

    dataset['y.Adm.Cr_l'] = np.log(dataset['y.Adm.Cr'])
    dataset['y.Admission.INR_l'] = np.log(dataset['y.Admission.INR'])

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns
    print(dataset.iloc[:, 5:].head(5))

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)

    # drop_list = ['x.HEP.C.RNA_N','y.Adm.WBC','x.GEND_F','x.GEND_M','x.HEP.C.RNA_P','x.Hep.B_N','x.Hep.B_P','y.Adm.Platelet','y.Adm.Alb','x.alcohol.use_HEAV<6M','x.alcohol.use_HEAV>6M','y.Adm.AST','y.Adm.ALP','x.alcohol.use_MOD','x.alcohol.use_SOC','y.AGE','y.BMI','y.Adm.Bili','y.Admission.INR','x.DM_Y','x.HTN_N','x.HTN_Y','y.Adm.Cr','y.Adm.Na','y.Adm.ALT','x.Cirrhosis_N','x.Cirrhosis_Y','x.DM_N']
    drop_list = ['x.HEP.C.RNA_N', 'y.Adm.WBC', 'x.GEND_F', 'x.GEND_M', 'x.HEP.C.RNA_P', 'x.Hep.B_N', 'x.Hep.B_P',
                 'y.Adm.Platelet', 'y.Adm.Alb', 'x.alcohol.use_HEAV<6M', 'x.alcohol.use_HEAV>6M', 'y.Adm.AST',
                 'y.Adm.ALP', 'x.alcohol.use_MOD', 'x.alcohol.use_SOC', 'y.AGE', 'y.BMI', 'y.Adm.Bili',
                 'x.DM_Y', 'x.HTN_N', 'x.HTN_Y', 'y.Adm.Na', 'y.Adm.ALT',
                 'x.DM_N']

    #for term in drop_list:
    #    X = X.drop([term], axis=1)

    acc_list = []
    kappa_list = []
    ap_list = []
    auc_list = []
    specificity_list = []
    sensitivity_list = []

    kf = KFold(n_splits=kfolds)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

    #for i in range(loopcount):
        # Split the dataset to trainand test data
    #    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

        # Build a neural network :
        NN_model = Sequential()
        NN_model.add(Dense(128, input_dim=30, activation='relu'))
        NN_model.add(Dense(512, activation='relu'))
        NN_model.add(Dense(512, activation='relu'))
        NN_model.add(Dense(512, activation='relu'))
        NN_model.add(Dense(512, activation='relu'))
        NN_model.add(Dense(512, activation='relu'))
        NN_model.add(Dense(256, activation='relu'))
        NN_model.add(Dense(1, activation='sigmoid'))
        NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #NN_model.compile(loss='mse', optimizer='sgd', metrics=[keras.metrics.TruePositives()])

        # checkpoint_name = 'Weights-{epoch:03d}-{val_accuracy:.5f}.hdf5'
        checkpoint_name = 'checkpoint.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        NN_model.fit(train_X, train_y, epochs=100, batch_size=30, validation_split=0.1, callbacks=callbacks_list)

        # wights_file = './Weights-016-0.88060.hdf5'  # choose the best checkpoint
        NN_model.load_weights(checkpoint_name)  # load it
        NN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #NN_model.compile(loss='mse', optimizer='sgd', metrics=[keras.metrics.TruePositives()])

        predictions = NN_model.predict(test_X)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        test_pred = rounded

        acc = metrics.accuracy_score(test_y, test_pred)
        acc_list.append(acc)
        kappa = metrics.cohen_kappa_score(test_y, test_pred)
        kappa_list.append(kappa)
        ap = metrics.average_precision_score(test_y, test_pred)
        ap_list.append(ap)
        auc = metrics.roc_auc_score(test_y, test_pred)
        auc_list.append(auc)

        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)
        sensitivity = tp / (tp + fn)
        sensitivity_list.append(sensitivity)

        # print(stdev(acc_list))
        # print(stdev(auc_list))
    return np.mean(acc_list), np.mean(kappa_list), np.mean(ap_list), np.mean(auc_list), np.mean(specificity_list), np.mean(sensitivity_list)

