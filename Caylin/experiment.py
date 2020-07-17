import csv
import numpy as np
import pickle

from progress.bar import Bar
from sklearn import ensemble, feature_selection, linear_model, metrics, pipeline, preprocessing, svm

from dataset import HepADataset


class StatusMessageBar(Bar):
    fill = '*'
    suffix = 'Run %(run)d: %(status_msg)s'
    status_msg = 'Idle'
    run = 0

    def set_run(self, _run):
        self.run = _run

    def set_status(self, msg):
        self.status_msg = msg


def add_stats(model, X, y, run_record):
    try:
        y_pred = model.predict(X)

        acc = metrics.accuracy_score(y, y_pred)
        ap = metrics.average_precision_score(y, y_pred)
        roc_auc = metrics.roc_auc_score(y, y_pred)
        kappa = metrics.cohen_kappa_score(y, y_pred)

        tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)

        run_record += [acc, ap, roc_auc, kappa, spec, sens]
    except Exception as e:
        print('add_stats exception: ', e)
        print(X)

feature_names = ['age', 'gender', 'bmi', 'hep_c_rna', 'hep_b', 'alch', 'cirr',
                 'adm_bili', 'adm_inr', 'adm_cr', 'adm_na', 'adm_alt', 'adm_ast',
                 'adm_alp', 'adm_plate', 'dm', 'htn', 'adm_alb', 'adm_wbc']

hepa_data_file = "hepa.csv"
results_file = "experiment-results.csv"
top_n_features = 5
use_select_k_best = True
runs = 1000
include_n_features = True
do_rf = True
do_lr = True
do_svm = True
do_full_data = True
rf_options = {
    'bootstrap': True,
    'min_samples_leaf': 3,
    'n_estimators': 300,
    'min_samples_split': 10,
    'max_features': 'sqrt',
    'max_depth': 6,
    'max_leaf_nodes': None,
}
lr_options = {
    'max_iter': 10000,
    'solver': 'liblinear',
}
svm_options = {
    'max_iter': 500000,
}

with open(results_file, 'w') as csvfile:
    resultswrite = csv.writer(csvfile)
    header = []
    if do_rf:
        header += ['rf_acc', 'rf_ap', 'rf_roc_auc', 'rf_kappa', 'rf_spec', 'rf_sens']
        if do_full_data:
            header += ['rf_all_data_acc', 'rf_all_data_ap', 'rf_all_data_roc_auc',
                       'rf_all_data_kappa', 'rf_all_data_spec', 'rf_all_data_sens']
    if do_lr:
        header += ['lr_acc', 'lr_ap', 'lr_roc_auc', 'lr_kappa', 'lr_spec', 'lr_sens']
        if do_full_data:
            header += ['lr_all_data_acc', 'lr_all_data_ap', 'lr_all_data_roc_auc',
                       'lr_all_data_kappa', 'lr_all_data_spec', 'lr_all_data_sens']
    if do_svm:
        header += ['svm_acc', 'svm_ap', 'svm_roc_auc', 'svm_kappa', 'svm_spec', 'svm_sens']
        if do_full_data:
            header += ['svm_all_data_acc', 'svm_all_data_ap', 'svm_all_data_roc_auc',
                       'svm_all_data_kappa', 'svm_all_data_spec', 'svm_all_data_sens']
    if include_n_features:
        if do_rf:
            header += [f"rf_{top_n_features}_acc", f"rf_{top_n_features}_ap", f"rf_{top_n_features}_roc_auc",
                       f"rf_{top_n_features}_kappa", f"rf_{top_n_features}_spec", f"rf_{top_n_features}_sens"]
            if do_full_data:
                header += [f"rf_all_data_{top_n_features}_acc", f"rf_all_data_{top_n_features}_ap",
                           f"rf_all_data_{top_n_features}_roc_auc", f"rf_all_data_{top_n_features}_kappa",
                           f"rf_all_data_{top_n_features}_spec", f"rf_all_data_{top_n_features}_sens"]
        if do_lr:
            header += [f"lr_{top_n_features}_acc", f"lr_{top_n_features}_ap", f"lr_{top_n_features}_roc_auc",
                       f"lr_{top_n_features}_kappa", f"lr_{top_n_features}_spec", f"lr_{top_n_features}_sens"]
            if do_full_data:
                header += [f"lr_all_data_{top_n_features}_acc", f"lr_all_data_{top_n_features}_ap",
                           f"lr_all_data_{top_n_features}_roc_auc", f"lr_all_data_{top_n_features}_kappa",
                           f"lr_all_data_{top_n_features}_spec", f"lr_all_data_{top_n_features}_sens"]
        if do_svm:
            header += [f"svm_{top_n_features}_acc", f"svm_{top_n_features}_ap", f"svm_{top_n_features}_roc_auc",
                       f"svm_{top_n_features}_kappa", f"svm_{top_n_features}_spec", f"svm_{top_n_features}_sens"]
            if do_full_data:
                header += [f"svm_all_data_{top_n_features}_acc", f"svm_all_data_{top_n_features}_ap",
                           f"svm_all_data_{top_n_features}_roc_auc", f"svm_all_data_{top_n_features}_kappa",
                           f"svm_all_data_{top_n_features}_spec", f"svm_all_data_{top_n_features}_sens"]
        header += feature_names
    resultswrite.writerow(header)
    experiments_per_run = (1 if do_rf else 0) + (1 if do_lr else 0) + (1 if do_svm else 0)
    if include_n_features:
        experiments_per_run = experiments_per_run * 2 + 1
    experiments_per_run += 1
    bar = StatusMessageBar('Running', max=(experiments_per_run * runs))
    run = 1
    while run <= runs:
        bar.set_run(run)
        bar.set_status(f"Starting run")
        dataset = HepADataset(hepa_data_file, initial_partition=.8)
        x_full, y_full = dataset.get_full_data()
        x_train, y_train = dataset.get_training()
        x_test, y_test = dataset.get_testing()
        bar.next()
        try:
            result_row = []
            rf_clf = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                            ensemble.RandomForestClassifier(**rf_options))
            lr_clf = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                            linear_model.LogisticRegression(**lr_options))
            svm_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))
            if do_rf:
                bar.set_status(f"Evaluating RandomForestClassifier")
                rf_clf.fit(x_train, y_train)
                add_stats(rf_clf, x_test, y_test, result_row)
                if do_full_data:
                    add_stats(rf_clf, x_full, y_full, result_row)
                bar.next()
            if do_lr:
                bar.set_status(f"Evaluating LogisticRegressionClassifier")
                lr_clf.fit(x_train, y_train)
                add_stats(lr_clf, x_test, y_test, result_row)
                if do_full_data:
                    add_stats(lr_clf, x_full, y_full, result_row)
                bar.next()
            if do_svm:
                bar.set_status(f"Evaluating SupportVectorMachineClassifier")
                svm_clf.fit(x_train, y_train)
                add_stats(svm_clf, x_test, y_test, result_row)
                if do_full_data:
                    add_stats(svm_clf, x_full, y_full, result_row)
                bar.next()
            if include_n_features:
                """bar.set_status(f"Finding top {top_n_features} features for this training set")
                if use_select_k_best:
                    fs = feature_selection.SelectKBest(k=top_n_features)
                    fs.fit(x_train, y_train)
                    best_feats = np.flip(np.array(fs.scores_).argsort())[:top_n_features]
                else:
                    feature_trainer = linear_model.LogisticRegressionCV(solver='liblinear', max_iter=100)
                    selector = feature_selection.RFE(feature_trainer, n_features_to_select=top_n_features, step=5)
                    selector = selector.fit(x_train, y_train)
                    best_feats = [i for i in range(len(selector.support_)) if selector.support_[i] == True]"""
                best_feats = [6, 9, 10, 11, 17]
                selected_feats = [True if i in best_feats else False for i in range(19)]
                dataset.set_columns(best_feats)
                x_full_limit_feats, y_full_limit_feats = dataset.get_full_data()
                x_train_limit_feats, y_train_limit_feats = dataset.get_training()
                x_test_limit_feats, y_test_limit_feats = dataset.get_testing()
                bar.next()
                rf_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            ensemble.RandomForestClassifier(**rf_options))
                lr_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            linear_model.LogisticRegression(**lr_options))
                svm_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))
                if do_rf:
                    bar.set_status(f"Starting Top {top_n_features} Feature RandomForestClassifier")
                    rf_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    add_stats(rf_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, result_row)
                    if do_full_data:
                        add_stats(rf_clf_limit_feats, x_full_limit_feats, y_full_limit_feats, result_row)
                    bar.next()
                if do_lr:
                    bar.set_status(f"Starting Top {top_n_features} Feature LogisticRegressionClassifier")
                    lr_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    add_stats(lr_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, result_row)
                    if do_full_data:
                        add_stats(lr_clf_limit_feats, x_full_limit_feats, y_full_limit_feats, result_row)
                    bar.next()
                if do_svm:
                    bar.set_status(f"Starting Top {top_n_features} Feature SupportVectorMachineClassifier")
                    svm_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    add_stats(svm_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, result_row)
                    if do_full_data:
                        add_stats(svm_clf_limit_feats, x_full_limit_feats, y_full_limit_feats, result_row)
                    bar.next()
                result_row += selected_feats
            resultswrite.writerow(result_row)
            run += 1
        except ValueError as e:
            print(f"Exception: {e}")
bar.set_status(f"Experiment complete")
bar.update()
bar.finish()
print(f"Results written to: {results_file}")
