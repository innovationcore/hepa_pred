import csv
# import matplotlib.pyplot as plt

from sklearn import ensemble, feature_selection, linear_model, metrics, pipeline, preprocessing, svm

from dataset import HepADataset

feature_names = ['age', 'gender', 'bmi', 'hep_c_rna', 'hep_b', 'alch', 'cirr',
                 'adm_bili', 'adm_inr', 'adm_cr', 'adm_na', 'adm_alt', 'adm_ast',
                 'adm_alp', 'adm_plate', 'dm', 'htn', 'adm_alb', 'adm_wbc']

runs = 400
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

with open('experiment-results.csv', 'w') as csvfile:
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
            header += ['rf_5_acc', 'rf_5_ap', 'rf_5_roc_auc', 'rf_5_kappa', 'rf_5_spec', 'rf_5_sens']
        if do_lr:
            header += ['lr_5_acc', 'lr_5_ap', 'lr_5_roc_auc', 'lr_5_kappa', 'lr_5_spec', 'lr_5_sens']
        if do_svm:
            header += ['svm_5_acc', 'svm_5_ap', 'svm_5_roc_auc', 'svm_5_kappa', 'svm_5_spec', 'svm_5_sens']
        header += feature_names
    resultswrite.writerow(header)
    run = 1
    while run <= runs:
        print(f"Run: {run}")
        if print_to_screen:
            print("Loading HEPA dataset")
        dataset = HepADataset('hepa.csv', fill_in_missing=False, initial_partition=.75)
        x_train, y_train = dataset.get_training()
        x_test, y_test = dataset.get_testing()
        if print_to_screen:
            print(f"Loaded {len(x_train)} training samples and {len(x_test)} testing samples")
        try:
            result_row = []
            rf_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.RandomForestClassifier(**rf_options))
            lr_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegressionCV(**lr_options))
            svm_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))

            if do_rf:
                print("Starting RandomForestClassifier")
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
                # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                if print_to_screen:
                    print(f"Testing Accuracy: {rf_test_acc*100:0.2f}%")
                    print(f"Kappa: {rf_test_kappa}")
                    print(f"AP: {rf_test_ap}")
                    print(f"ROC AUC: {rf_test_roc_auc}")
                    print(f"Specificity: {rf_test_spec}")
                    print(f"Sensitivity: {rf_test_sens}")
                result_row += [rf_test_acc, rf_test_ap, rf_test_roc_auc, rf_test_kappa, rf_test_spec, rf_test_sens]

            if do_lr:
                print("Starting LogisticRegressionClassifier")
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
                # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                if print_to_screen:
                    print(f"Testing Accuracy: {lr_test_acc*100:0.2f}%")
                    print(f"Kappa: {lr_test_kappa}")
                    print(f"AP: {lr_test_ap}")
                    print(f"ROC AUC: {lr_test_roc_auc}")
                    print(f"Specificity: {lr_test_spec}")
                    print(f"Sensitivity: {lr_test_sens}")
                result_row += [lr_test_acc, lr_test_ap, lr_test_roc_auc, lr_test_kappa, lr_test_spec, lr_test_sens]

            if do_svm:
                print("Starting SupportVectorMachineClassifier")
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
                # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                if print_to_screen:
                    print(f"Testing Accuracy: {svm_test_acc * 100:0.2f}%")
                    print(f"Kappa: {svm_test_kappa}")
                    print(f"AP: {svm_test_ap}")
                    print(f"ROC AUC: {svm_test_roc_auc}")
                    print(f"Specificity: {svm_test_spec}")
                    print(f"Sensitivity: {svm_test_sens}")
                result_row += [svm_test_acc, svm_test_ap, svm_test_roc_auc, svm_test_kappa, svm_test_spec, svm_test_sens]

            print("Starting SupportVectorMachineClassifier")
            nb_clf.fit(x_train, y_train)
            train_pred = nb_clf.predict(x_train)
            print(f"Training Accuracy: {metrics.accuracy_score(y_train, train_pred) * 100:0.2f}%")
            test_pred = nb_clf.predict(x_test)
            nb_test_acc = metrics.accuracy_score(y_test, test_pred)
            nb_test_kappa = metrics.cohen_kappa_score(y_test, test_pred)
            nb_test_ap = metrics.average_precision_score(y_test, test_pred)
            nb_test_roc_auc = metrics.roc_auc_score(y_test, test_pred)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
            nb_test_spec = tn / (tn + fp)
            nb_test_sens = tp / (tp + fn)
            # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
            print(f"Testing Accuracy: {nb_test_acc * 100:0.2f}%")
            print(f"Kappa: {nb_test_kappa}")
            print(f"AP: {nb_test_ap}")
            print(f"ROC AUC: {nb_test_roc_auc}")
            print(f"Specificity: {nb_test_spec}")
            print(f"Sensitivity: {nb_test_sens}")

            if include_5_features:
                print("Finding top 5 features")
                feature_trainer = linear_model.LogisticRegressionCV(solver='liblinear', max_iter=100)
                selector = feature_selection.RFE(feature_trainer, n_features_to_select=5, step=5)
                selector = selector.fit(x_train, y_train)
                features_to_use = [i for i in range(len(selector.support_)) if selector.support_[i] == True]
                print("Features:", ",".join([feature_names[i] for i in features_to_use]))
                dataset.set_columns(features_to_use)
                x_train_limit_feats, y_train_limit_feats = dataset.get_training()
                x_test_limit_feats, y_test_limit_feats = dataset.get_testing()

                rf_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            ensemble.RandomForestClassifier(**rf_options))
                lr_clf_limit_feats = pipeline.make_pipeline(preprocessing.StandardScaler(),
                                                            linear_model.LogisticRegressionCV(**lr_options))
                svm_5_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(**svm_options))

                if do_rf:
                    print("Starting Top 5 Feature RandomForestClassifier")
                    rf_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    train_pred_limit_feats = rf_clf_limit_feats.predict(x_train_limit_feats)
                    if print_to_screen:
                        print(f"Training Accuracy: {metrics.accuracy_score(y_train_limit_feats, train_pred_limit_feats)*100:0.2f}%")
                    test_pred_limit_feats = rf_clf_limit_feats.predict(x_test_limit_feats)
                    rf_5_test_acc = metrics.accuracy_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_kappa = metrics.cohen_kappa_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_ap = metrics.average_precision_score(y_test_limit_feats, test_pred_limit_feats)
                    rf_5_test_roc_auc = metrics.roc_auc_score(y_test_limit_feats, test_pred_limit_feats)
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test_limit_feats, test_pred_limit_feats).ravel()
                    rf_5_test_spec = tn / (tn + fp)
                    rf_5_test_sens = tp / (tp + fn)
                    # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                    if print_to_screen:
                        print(f"Testing Accuracy: {rf_5_test_acc*100:0.2f}%")
                        print(f"Kappa: {rf_5_test_kappa}")
                        print(f"AP: {rf_5_test_ap}")
                        print(f"ROC AUC: {rf_5_test_roc_auc}")
                        print(f"Specificity: {rf_5_test_spec}")
                        print(f"Sensitivity: {rf_5_test_sens}")
                    result_row += [rf_5_test_acc, rf_5_test_ap, rf_5_test_roc_auc,
                                   rf_5_test_kappa, rf_5_test_spec, rf_5_test_sens]

                if do_lr:
                    print("Starting Top 5 Feature LogisticRegressionClassifier")
                    lr_clf_limit_feats.fit(x_train_limit_feats, y_train_limit_feats)
                    train_pred_limit_feats = lr_clf_limit_feats.predict(x_train_limit_feats)
                    if print_to_screen:
                        print(f"Training Accuracy: {metrics.accuracy_score(y_train_limit_feats, train_pred_limit_feats)*100:0.2f}%")
                    test_pred_limit_feats = lr_clf_limit_feats.predict(x_test_limit_feats)
                    lr_5_test_acc = metrics.accuracy_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_kappa = metrics.cohen_kappa_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_ap = metrics.average_precision_score(y_test_limit_feats, test_pred_limit_feats)
                    lr_5_test_roc_auc = metrics.roc_auc_score(y_test_limit_feats, test_pred_limit_feats)
                    tn, fp, fn, tp = metrics.confusion_matrix(y_test_limit_feats, test_pred_limit_feats).ravel()
                    lr_5_test_spec = tn / (tn + fp)
                    lr_5_test_sens = tp / (tp + fn)
                    # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                    if print_to_screen:
                        print(f"Testing Accuracy: {lr_5_test_acc*100:0.2f}%")
                        print(f"Kappa: {lr_5_test_kappa}")
                        print(f"AP: {lr_5_test_ap}")
                        print(f"ROC AUC: {lr_5_test_roc_auc}")
                        print(f"Specificity: {lr_5_test_spec}")
                        print(f"Sensitivity: {lr_5_test_sens}")
                    result_row += [lr_5_test_acc, lr_5_test_ap, lr_5_test_roc_auc,
                                   lr_5_test_kappa, lr_5_test_spec, lr_5_test_sens]

                if do_svm:
                    print("Starting Top 5 Feature SupportVectorMachineClassifier")
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
                    # print(f"Report: {metrics.classification_report(y_test, test_pred)}")
                    if print_to_screen:
                        print(f"Testing Accuracy: {svm_test_acc * 100:0.2f}%")
                        print(f"Kappa: {svm_5_test_kappa}")
                        print(f"AP: {svm_5_test_ap}")
                        print(f"ROC AUC: {svm_5_test_roc_auc}")
                        print(f"Specificity: {svm_5_test_spec}")
                        print(f"Sensitivity: {svm_5_test_sens}")
                    result_row += [svm_5_test_acc, svm_5_test_ap, svm_5_test_roc_auc,
                                   svm_5_test_kappa, svm_5_test_spec, svm_5_test_sens]

                """rf_pr = metrics.plot_precision_recall_curve(rf_clf, x_test, y_test)
                metrics.plot_precision_recall_curve(rf_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, 
                ax=rf_pr.ax_, name="RandomForest5Feat")
                metrics.plot_precision_recall_curve(lr_clf, x_test, y_test, ax=rf_pr.ax_)
                metrics.plot_precision_recall_curve(lr_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, 
                ax=rf_pr.ax_, name="LogisticRegression5Feat")
                plt.title('Precision Recall Curve')
                plt.show()
                rf_roc = metrics.plot_roc_curve(rf_clf, x_test, y_test)
                metrics.plot_roc_curve(rf_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, ax=rf_roc.ax_, 
                name="RandomForest5Feat")
                metrics.plot_roc_curve(lr_clf, x_test, y_test, ax=rf_roc.ax_)
                metrics.plot_roc_curve(lr_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, ax=rf_roc.ax_, 
                name="LogisticRegression5Feat")
                plt.title('Reciever-Operator Curve')
                plt.show()
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
                axes[0][0].set_title('RandomForestClassifier')
                axes[0][1].set_title('RandomForest5Feat')
                axes[1][0].set_title('LogisticRegressionClassifier')
                axes[1][1].set_title('LogisticRegression5Feat')
                metrics.plot_confusion_matrix(rf_clf, x_test, y_test, display_labels=['NOT-BAD', 'BAD'], ax=axes[0][0])
                metrics.plot_confusion_matrix(rf_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, 
                display_labels=['NOT-BAD', 'BAD'], ax=axes[0][1])
                metrics.plot_confusion_matrix(lr_clf, x_test, y_test, display_labels=['NOT-BAD', 'BAD'], ax=axes[1][0])
                metrics.plot_confusion_matrix(lr_clf_limit_feats, x_test_limit_feats, y_test_limit_feats, 
                display_labels=['NOT-BAD', 'BAD'], ax=axes[1][1])
                plt.suptitle('Confusion Matrices')
                plt.show()"""
            if include_5_features:
                result_row += selector.support_.tolist()
            resultswrite.writerow(result_row)
            run += 1
        except ValueError as e:
            print(x_train)
            print(y_train)
            print(x_test)
            print(y_test)
            print(f"Exception: {e}")
