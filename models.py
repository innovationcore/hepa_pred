from statistics import stdev

import keras
from sklearn.model_selection import KFold
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

