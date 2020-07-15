import csv
import numpy as np

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


feature_names = ['age', 'gender', 'bmi', 'hep_c_rna', 'hep_b', 'alch', 'cirr',
                 'adm_bili', 'adm_inr', 'adm_cr', 'adm_na', 'adm_alt', 'adm_ast',
                 'adm_alp', 'adm_plate', 'dm', 'htn', 'adm_alb', 'adm_wbc']

hepa_data_file = "../../hepa.csv"
results_file = "experiment-results.csv"
top_n_features = 3
use_select_k_best = True
runs = 100
print_to_screen = False
include_5_features = True
do_rf = True
do_lr = True
do_svm = True
rf_options = {
    'bootstrap': True,
    'min_samples_leaf': 3,
    'n_estimators': 50,
    'min_samples_split': 10,
    'max_features': 'sqrt',
    'max_depth': 6,
    'max_leaf_nodes': None,
}
lr_options = {
    'cv': 10,
    'max_iter': 1000,
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
    if do_lr:
        header += ['lr_acc', 'lr_ap', 'lr_roc_auc', 'lr_kappa', 'lr_spec', 'lr_sens']
    if do_svm:
        header += ['svm_acc', 'svm_ap', 'svm_roc_auc', 'svm_kappa', 'svm_spec', 'svm_sens']
    if include_5_features:
        if do_rf:
            header += [f"rf_{top_n_features}_acc", f"rf_{top_n_features}_ap", f"rf_{top_n_features}_roc_auc",
                       f"rf_{top_n_features}_kappa", f"rf_{top_n_features}_spec", f"rf_{top_n_features}_sens"]
        if do_lr:
            header += [f"lr_{top_n_features}_acc", f"lr_{top_n_features}_ap", f"lr_{top_n_features}_roc_auc",
                       f"lr_{top_n_features}_kappa", f"lr_{top_n_features}_spec", f"lr_{top_n_features}_sens"]
        if do_svm:
            header += [f"svm_{top_n_features}_acc", f"svm_{top_n_features}_ap", f"svm_{top_n_features}_roc_auc",
                       f"svm_{top_n_features}_kappa", f"svm_{top_n_features}_spec", f"svm_{top_n_features}_sens"]
        header += feature_names
    resultswrite.writerow(header)
    experiments_per_run = (1 if do_rf else 0) + (1 if do_lr else 0) + (1 if do_svm else 0)
    if include_5_features:
        experiments_per_run = experiments_per_run * 2 + 1
    experiments_per_run += 1
    bar = StatusMessageBar('Running', max=(experiments_per_run * runs))
    run = 1
    while run <= runs:
        bar.set_run(run)
        bar.set_status(f"Starting run")
        if print_to_screen:
            print("Loading HEPA dataset")
        dataset = HepADataset(hepa_data_file, initial_partition=.8)
        x_train, y_train = dataset.get_training()
        x_test, y_test = dataset.get_testing()
        if print_to_screen:
            print(f"Loaded {len(x_train)} training samples and {len(x_test)} testing samples")
        bar.next()
        try:
            result_row = []
            rf_clf = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                            ensemble.RandomForestClassifier(**rf_options))
            lr_clf = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                            linear_model.LogisticRegressionCV(**lr_options))
            svm_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))
            if do_rf:
                bar.set_status(f"Evaluating RandomForestClassifier")
                rf_clf.fit(x_train, y_train)
                train_pred = rf_clf.predict(x_train)
                if print_to_screen:
                    print(f"Training Accuracy: {metrics.accuracy_score(y_train, train_pred)*100:0.2f}%")
                test_pred = rf_clf.predict(x_test)
                rf_test_acc = metrics.accuracy_score(y_test, test_pred)
                rf_test_kappa = metrics.cohen_kappa_score(y_test, test_pred)
                rf_test_ap = metrics.average_precision_score(y_test, test_pred)
                rf_test_roc_auc = metrics.roc_auc_score(y_test, test_pred)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
                rf_test_spec = tn / (tn + fp)
                rf_test_sens = tp / (tp + fn)
                if print_to_screen:
                    print(f"Testing Accuracy: {rf_test_acc*100:0.2f}%")
                    print(f"Kappa: {rf_test_kappa}")
                    print(f"AP: {rf_test_ap}")
                    print(f"ROC AUC: {rf_test_roc_auc}")
                    print(f"Specificity: {rf_test_spec}")
                    print(f"Sensitivity: {rf_test_sens}")
                result_row += [rf_test_acc, rf_test_ap, rf_test_roc_auc, rf_test_kappa, rf_test_spec, rf_test_sens]
                bar.next()
            if do_lr:
                bar.set_status(f"Evaluating LogisticRegressionClassifier")
                lr_clf.fit(x_train, y_train)
                train_pred = lr_clf.predict(x_train)
                if print_to_screen:
                    print(f"Training Accuracy: {metrics.accuracy_score(y_train, train_pred)*100:0.2f}%")
                test_pred = lr_clf.predict(x_test)
                lr_test_acc = metrics.accuracy_score(y_test, test_pred)
                lr_test_kappa = metrics.cohen_kappa_score(y_test, test_pred)
                lr_test_ap = metrics.average_precision_score(y_test, test_pred)
                lr_test_roc_auc = metrics.roc_auc_score(y_test, test_pred)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
                lr_test_spec = tn / (tn + fp)
                lr_test_sens = tp / (tp + fn)
                if print_to_screen:
                    print(f"Testing Accuracy: {lr_test_acc*100:0.2f}%")
                    print(f"Kappa: {lr_test_kappa}")
                    print(f"AP: {lr_test_ap}")
                    print(f"ROC AUC: {lr_test_roc_auc}")
                    print(f"Specificity: {lr_test_spec}")
                    print(f"Sensitivity: {lr_test_sens}")
                result_row += [lr_test_acc, lr_test_ap, lr_test_roc_auc, lr_test_kappa, lr_test_spec, lr_test_sens]
                bar.next()
            if do_svm:
                bar.set_status(f"Evaluating SupportVectorMachineClassifier")
                svm_clf.fit(x_train, y_train)
                train_pred = svm_clf.predict(x_train)
                if print_to_screen:
                    print(f"Training Accuracy: {metrics.accuracy_score(y_train, train_pred) * 100:0.2f}%")
                test_pred = svm_clf.predict(x_test)
                svm_test_acc = metrics.accuracy_score(y_test, test_pred)
                svm_test_kappa = metrics.cohen_kappa_score(y_test, test_pred)
                svm_test_ap = metrics.average_precision_score(y_test, test_pred)
                svm_test_roc_auc = metrics.roc_auc_score(y_test, test_pred)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
                svm_test_spec = tn / (tn + fp)
                svm_test_sens = tp / (tp + fn)
                if print_to_screen:
                    print(f"Testing Accuracy: {svm_test_acc * 100:0.2f}%")
                    print(f"Kappa: {svm_test_kappa}")
                    print(f"AP: {svm_test_ap}")
                    print(f"ROC AUC: {svm_test_roc_auc}")
                    print(f"Specificity: {svm_test_spec}")
                    print(f"Sensitivity: {svm_test_sens}")
                result_row += [svm_test_acc, svm_test_ap, svm_test_roc_auc,
                               svm_test_kappa, svm_test_spec, svm_test_sens]
                bar.next()
            if include_5_features:
                bar.set_status(f"Finding top {top_n_features} features for this training set")
                if use_select_k_best:
                    fs = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=3)
                    fs.fit(x_train, y_train)
                    best_feats = np.flip(np.array(fs.scores_).argsort())[:top_n_features]
                else:
                    feature_trainer = linear_model.LogisticRegressionCV(solver='liblinear', max_iter=100)
                    selector = feature_selection.RFE(feature_trainer, n_features_to_select=top_n_features, step=5)
                    selector = selector.fit(x_train, y_train)
                    best_feats = [i for i in range(len(selector.support_)) if selector.support_[i] == True]
                selected_feats = [True if i in best_feats else False for i in range(19)]
                dataset.set_columns(best_feats)
                x_train_limit_feats, y_train_limit_feats = dataset.get_training()
                x_test_limit_feats, y_test_limit_feats = dataset.get_testing()
                bar.next()
                rf_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            ensemble.RandomForestClassifier(**rf_options))
                lr_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            linear_model.LogisticRegressionCV(**lr_options))
                svm_5_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))
                if do_rf:
                    bar.set_status(f"Starting Top {top_n_features} Feature RandomForestClassifier")
                    rf_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    train_pred_limit_feats = rf_clf_limit_feats.predict(x_train_limit_feats)
                    if print_to_screen:
                        print(
                            f"Training Accuracy: "
                            f"{metrics.accuracy_score(y_train_limit_feats, train_pred_limit_feats)*100:0.2f}%"
                        )
                    test_pred_limit_feats = rf_clf_limit_feats.predict(x_test_limit_feats)
                    rf_5_test_acc = metrics.accuracy_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_kappa = metrics.cohen_kappa_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_ap = metrics.average_precision_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_roc_auc = metrics.roc_auc_score(y_test_limit_feats, test_pred_limit_feats)
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test_limit_feats, test_pred_limit_feats).ravel()
                    rf_5_test_spec = tn / (tn + fp)
                    rf_5_test_sens = tp / (tp + fn)
                    if print_to_screen:
                        print(f"Testing Accuracy: {rf_5_test_acc*100:0.2f}%")
                        print(f"Kappa: {rf_5_test_kappa}")
                        print(f"AP: {rf_5_test_ap}")
                        print(f"ROC AUC: {rf_5_test_roc_auc}")
                        print(f"Specificity: {rf_5_test_spec}")
                        print(f"Sensitivity: {rf_5_test_sens}")
                    result_row += [rf_5_test_acc, rf_5_test_ap, rf_5_test_roc_auc,
                                   rf_5_test_kappa, rf_5_test_spec, rf_5_test_sens]
                    bar.next()
                if do_lr:
                    bar.set_status(f"Starting Top {top_n_features} Feature LogisticRegressionClassifier")
                    lr_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    train_pred_limit_feats = lr_clf_limit_feats.predict(x_train_limit_feats)
                    if print_to_screen:
                        print(f"Training Accuracy: "
                              f"{metrics.accuracy_score(y_train_limit_feats, train_pred_limit_feats)*100:0.2f}%"
                              )
                    test_pred_limit_feats = lr_clf_limit_feats.predict(x_test_limit_feats)
                    lr_5_test_acc = metrics.accuracy_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_kappa = metrics.cohen_kappa_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_ap = metrics.average_precision_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_roc_auc = metrics.roc_auc_score(y_test_limit_feats, test_pred_limit_feats)
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test_limit_feats, test_pred_limit_feats).ravel()
                    lr_5_test_spec = tn / (tn + fp)
                    lr_5_test_sens = tp / (tp + fn)
                    if print_to_screen:
                        print(f"Testing Accuracy: {lr_5_test_acc*100:0.2f}%")
                        print(f"Kappa: {lr_5_test_kappa}")
                        print(f"AP: {lr_5_test_ap}")
                        print(f"ROC AUC: {lr_5_test_roc_auc}")
                        print(f"Specificity: {lr_5_test_spec}")
                        print(f"Sensitivity: {lr_5_test_sens}")
                    result_row += [lr_5_test_acc, lr_5_test_ap, lr_5_test_roc_auc,
                                   lr_5_test_kappa, lr_5_test_spec, lr_5_test_sens]
                    bar.next()
                if do_svm:
                    bar.set_status(f"Starting Top {top_n_features} Feature SupportVectorMachineClassifier")
                    svm_5_clf.fit(x_train, y_train)
                    train_pred = svm_5_clf.predict(x_train)
                    if print_to_screen:
                        print(f"Training Accuracy: {metrics.accuracy_score(y_train, train_pred) * 100:0.2f}%")
                    test_pred = svm_5_clf.predict(x_test)
                    svm_5_test_acc = metrics.accuracy_score(y_test, test_pred)
                    svm_5_test_kappa = metrics.cohen_kappa_score(y_test, test_pred)
                    svm_5_test_ap = metrics.average_precision_score(y_test, test_pred)
                    svm_5_test_roc_auc = metrics.roc_auc_score(y_test, test_pred)
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
                    svm_5_test_spec = tn / (tn + fp)
                    svm_5_test_sens = tp / (tp + fn)
                    if print_to_screen:
                        print(f"Testing Accuracy: {svm_test_acc * 100:0.2f}%")
                        print(f"Kappa: {svm_5_test_kappa}")
                        print(f"AP: {svm_5_test_ap}")
                        print(f"ROC AUC: {svm_5_test_roc_auc}")
                        print(f"Specificity: {svm_5_test_spec}")
                        print(f"Sensitivity: {svm_5_test_sens}")
                    result_row += [svm_5_test_acc, svm_5_test_ap, svm_5_test_roc_auc,
                                   svm_5_test_kappa, svm_5_test_spec, svm_5_test_sens]
                    bar.next()
            if include_5_features:
                result_row += selected_feats
            resultswrite.writerow(result_row)
            run += 1
        except ValueError as e:
            print(x_train)
            print(y_train)
            print(x_test)
            print(y_test)
            print(f"Exception: {e}")
bar.set_status(f"Experiment complete")
bar.update()
bar.finish()
print(f"Results written to: {results_file}")
