from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
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
    RF_predictions = RF_model.predict(test_X)
    score = accuracy_score(test_y, RF_predictions)

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

    return score

def getrandomforestave(file_path, loopcount):
    # Read in data and display first 5 rows
    dataset = pd.read_csv(file_path)

    #dataset = dataset.fillna(dataset.mean())
    dataset = dataset.fillna(dataset.mode().iloc[0])
    print(dataset.columns[dataset.isna().any()].tolist())

    print("pre NA drop: " + str(dataset.shape))
    #dataset = dataset.dropna()
    print("post NA drop: " + str(dataset.shape))

    # One-hot encode the data using pandas get_dummies
    dataset = pd.get_dummies(dataset)
    # Display the first 5 rows of the last 12 columns

    y = dataset['z.bad_Y']
    X = dataset.drop(['z.bad_Y'], axis=1)
    X = X.drop(['z.bad_N'], axis=1)
    X = X.drop(['y.BMI'], axis=1)

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

