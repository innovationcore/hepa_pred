from models import getrandomforest, getnn, getrandomforestave

file_path = "/Users/cody/Downloads/hepa.csv"
loopcount = 1000

ave, max, min = getrandomforestave(file_path, loopcount)
print("RF LoopCount: "  + str(loopcount))
print("RF AVE: "  + str(ave))
print("RF MAX: "  + str(max))
print("RF MIN: "  + str(min))

rf_score = getrandomforest(file_path)
print("Random Forest Single Run Score: "  + str(rf_score))

nn_score = getnn(file_path)
print("Neural Network Score: "  + str(nn_score))



