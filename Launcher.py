from models import getrandomforest, getnn, getrandomforestave, getrandomforeststats_all, getrandomforeststats_3, \
    getnnstats, getrandomforeststats_fold, getrandomforeststats_fold_cv, getsample, testmodel

file_path = "hepaa.csv"
loopcount = 1000

'''
ave, max, min = getrandomforestave(file_path, loopcount)
print("RF LoopCount: "  + str(loopcount))
print("RF AVE: "  + str(ave))
print("RF MAX: "  + str(max))
print("RF MIN: "  + str(min))
'''

#testmodel(file_path)

#getsample(file_path)

#acc, kappa, ap, auc, specificity, sensitivity = getrandomforeststats_all(file_path, loopcount)

#acc, kappa, ap, auc, specificity, sensitivity = getrandomforeststats_3(file_path, loopcount)
#print("acc: " + str(acc) + " ap: " + str(ap) + " auc:" + str(auc) + " kappa: " + str(kappa) + " specificity: " + str(specificity) + " sensitivity: " + str(sensitivity))

#used for model
getrandomforeststats_fold_cv(file_path, 500)
#acc, kappa, ap, auc, specificity, sensitivity = getrandomforeststats_fold_cv(file_path, 5)
#print("acc: " + str(acc) + " ap: " + str(ap) + " auc:" + str(auc) + " kappa: " + str(kappa) + " specificity: " + str(specificity) + " sensitivity: " + str(sensitivity))

#acc, kappa, ap, auc, specificity, sensitivity = getnnstats(file_path, 5)
#print("acc: " + str(acc) + " ap: " + str(ap) + " auc:" + str(auc) + " kappa: " + str(kappa) + " specificity: " + str(specificity) + " sensitivity: " + str(sensitivity))

#getrandomforest(file_path)

#rf_score = getrandomforest(file_path)
#print("Random Forest Single Run Score: "  + str(rf_score))

#nn_score = getnn(file_path)
#print("Neural Network Score: "  + str(nn_score))



