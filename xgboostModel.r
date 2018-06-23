## xgboost

library(xgboost)
trainDataXgBoost <- trainData
validationXgBoost <- validationData
testDataXgBoost <- testData

sum(is.na(trainDataXgBoost))

trainDataXgBoost$classlabel <- ifelse(trainDataXgBoost$target == 0, "solvent", "bankrupt")
validationXgBoost$classlabel <- ifelse(validationXgBoost$target == 0, "solvent", "bankrupt")


train_Matrix <- xgb.DMatrix(data = as.matrix(trainDataXgBoost[,!(names(trainDataXgBoost) %in% c("target","classlabel"))]), 
                            label = as.matrix(trainDataXgBoost[,(names(trainDataXgBoost) %in% c("target"))]))

validate_Matrix <- xgb.DMatrix(data = as.matrix(validationXgBoost[,!(names(validationXgBoost) %in% c("target","classlabel"))]), 
                               label = as.matrix(validationXgBoost[,(names(validationXgBoost) %in% c("target"))]))

caret::modelLookup("xgbTree")


xgb_model_basic <- xgboost(data = train_Matrix, max.depth = 2, eta = 1, nthread = 2, nround = 400, objective = "binary:logistic", verbose = 1, early_stopping_rounds = 10)

basic_validation_preds <- predict(xgb_model_basic,validate_Matrix)
basic_validation_predLabels <- ifelse(basic_validation_preds > 0.7, 1, 0)

confMatrixValidate <- caret::confusionMatrix(basic_validation_predLabels, validationXgBoost$target, mode="prec_recall")
confMatrixValidate$byClass[[7]]
## prediction on test
test_Matrix
basic_test_preds <- predict(xgb_model_basic, as.matrix(testDataXgBoost),type="response")
basic_test_predvalue <- ifelse(basic_test_preds > 0.3, 1,0)
outputTestXgBoost <- data.frame(testData$ID,basic_test_predvalue)
colnames(outputTestXgBoost) <- c("ID","prediction")
table(outputTestXgBoost$prediction)
write.csv(outputTestXgBoost,"SampleSubmission_xgBasic.csv")

## plotting variable importance
variableimpMatrix <- xgb.importance(feature_names = colnames(trainDataXgBoost), model = xgb_model_basic)
plot(variableimpMatrix)

# tuning xgboost

sampling_strategy <- caret::trainControl(method = "repeatedcv", number = 5, repeats = 2, verboseIter = F, allowParallel = T)

param_grid <- expand.grid(.nrounds = 150, .max_depth = c(2, 4, 6,8), .eta = c(0.1, 0.3),
                          .gamma = c(0.6, 0.5, 0.3), .colsample_bytree = c(0.6, 0.4),
                          .min_child_weight = 1, .subsample = c(0.5, 0.6, 0.7))

xgb_tuned_model <- caret::train(x = trainDataXgBoost[ , !(names(trainDataXgBoost) %in% c("classlabel", "target"))], 
                         y = trainDataXgBoost[ , names(trainDataXgBoost) %in% c("classlabel")], 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         tuneGrid = param_grid)

xgb_tuned_model$bestTune
plot(xgb_tuned_model)

tuned_params_validation_preds <- predict(xgb_tuned_model, validationXgBoost[ , !(names(validationData) %in% c("classlabel", "class"))])
caret::confusionMatrix(tuned_params_validation_preds,validationXgBoost$target, mode="prec_recall")

tuned_test_preds <- predict(xgb_tuned_model,as.matrix(testDataXgBoost))
testOutputXgBoostTuned <- data.frame(testDataXgBoost$ID,tuned_test_preds)
colnames(testOutputXgBoostTuned) <- c("ID","prediction")
table(testOutputXgBoostTuned$prediction)
write.csv(testOutputXgBoostTuned,"SampleSubmission_xgTuned.csv")
