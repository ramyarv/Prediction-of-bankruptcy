rm(list = ls(all=TRUE))
setwd("C:\\Users\\rvadamala\\Documents\\insofe\\CUTE_CS7305C")
dir()

## Reading the data
inputData <-  read.csv("train.csv",header = T, na.strings = "NA")
testData <- read.csv("test.csv",header=T,na.strings = "NA")

str(inputData)
summary(inputData)

## Understanding of data

#1) ID is unique column, make the row names as ID
rownames(inputData) <- inputData$ID 
inputData$ID <-  NULL

#2) target is categorical , lets change that to factor datatype
inputData$target <- as.factor(as.character(inputData$target))

#3) Mean is very much far from Median for most of the attributes, that implies existence of outliers
### Let's look into it .. how many outliers are there in each column 

##----------------------------------------------------------##
##      Function to find percentage of Outliers in data     ##
##----------------------------------------------------------##


findNumOfOutliers <- function(x) {
  noOfColumns <- ncol(x)
  count <- c()
  for(col in 1:noOfColumns) {
    outliers <- boxplot.stats(x[,col])$out
    count <- c(count,length(outliers))
    
  }
  count <- ((count*100)/nrow(x))
  names(count) <- colnames(x)
  count <- sort(count,decreasing = T)
  
  return(count)
}

outlierCount <- findNumOfOutliers(inputData[,-ncol(inputData)])
print(outlierCount)

## Observations : It is observed that there are lot of outliers in this data.. but when we look into the business problem
## it is to found the bankruptcy of the firms..may be these are the values which will make sense to show a bankrupted company
## so, lets keep it for now.. and may be before modelling.. we may need to impose a method to do feature selection and 
## remove the insignificant variables.

## Any Missing values ??

sum(is.na(inputData))

##----------------------------------------------------------##
##    Function to find columns with more than 20% NAs       ##
##----------------------------------------------------------##

findRemovableNaColumns <- function(x){
  colSums <- (colSums(is.na(x)) / nrow(x) ) * 100
  names(colSums) <- colnames(x)
  colSums <- sort(colSums,decreasing = T)
  colNames <- names(colSums[colSums > 20])
  return (colNames)
}

naColsToBeRemoved <- findRemovableNaColumns(inputData[,-ncol(inputData)])
inputData <- inputData[,!(names(inputData) %in% naColsToBeRemoved)]
testData <- testData[,!(names(testData) %in% naColsToBeRemoved)]

## Imputing the remaining columns with centralImputation 
## Reason : centralImputation works with median, which is less affected by outliers

inputData <- DMwR::centralImputation(inputData)
testData <- DMwR::centralImputation(testData)
summary(inputData)
sum(is.na(inputData))



sum(is.na(testData))

## Handling class imbalance using ROSE package
plot(inputData$target)
## seems like class 0 is dominating in the data.
## lets try ot balance the data using synthetic sampling strategy by ROSE package

prop.table(table(inputData$target))
inputDataRose <- ROSE::ROSE(target~.,inputData,seed = 777,p = 0.20)$data
??ROSE::ROSE
prop.table(table(inputDataRose$target))
?ROSE::ROSE

##SMOTE??
inputDataSmote <- DMwR::SMOTE(target ~ ., inputData, perc.over = 30, k=5)
prop.table(table(inputDataSmote$target))
#set.seed(777)
#trainRows <- caret::createDataPartition(inputDataSmote$target,p=0.7,list=F)
#trainData <- inputDataSmote[trainRows,]
#validationData <- inputDataSmote[-trainRows,]

#sum(is.na(trainData))
## correlation plot
corrplot::corrplot(cor(inputDataRose[,!(names(inputDataRose)%in% c("target"))]),method="shade",type="full")

set.seed(777)

##train/validation split
trainRows <- caret::createDataPartition(inputData$target,p=0.7,list=F)
trainData <- inputData[trainRows,]
validationData <- inputData[-trainRows,]

##Standardization
stdObj <- caret::preProcess(trainData[,-ncol(trainData)], method = c("center","scale"))
trainData <- predict(stdObj,trainData)
validationData <- predict(stdObj,validationData)
testData <- predict(stdObj,testData)

## C5.0 decision tree
#install.packages("C50")
c50Tree <- C50::C5.0(target~.,trainData)
save(c50Tree,file="c50_model.rda")

summary(c50Tree)
plot(c50Tree)
c50Rules <- C50::C5.0(target~.,trainData, rules=T)
summary(c50Rules)

## predicting validation and test data
c5validationPreds <- predict(c50Tree, validationData)
caret::confusionMatrix(c5validationPreds, validationData$target, mode = "prec_recall")
# test data
c5testPreds <- predict(c50Tree,testData)
outputTest <- data.frame(testData$ID,c5testPreds)
colnames(outputTest) <- c("ID","prediction")
write.csv(outputTest,file = "SampleSubmission_C50.csv")
## DecisionTree - 54 %

c5_imp_values <- C50::C5imp(c50Tree)
c5_imp_attr <-rownames(c5_imp_values)[c5_imp_values$Overall>1]
print(c5_imp_attr)

## lets use it in our imp attr
traindata_c5imp <- trainData[,(names(trainData) %in% c(c5_imp_attr,"target"))]
names(traindata_c5imp)

## prediction with only imp attributes - c5
c50Tree_Imp <- C50::C5.0(target~.,traindata_c5imp)

c5impvalidationPreds <- predict(c50Tree_Imp, validationData)
caret::confusionMatrix(c5impvalidationPreds, validationData$target,mode = "prec_recall")
# test data
c5imptestPreds <- predict(c50Tree_Imp,testData)
outputTestImp <- data.frame(testData$ID,c5imptestPreds)
colnames(outputTestImp) <- c("ID","prediction")
write.csv(outputTestImp,file = "SampleSubmission_C50Imp.csv")

predictOnTest <- function(mod){
  testPreds <- predict(mod,testData,type="response")
  table(testPreds)
  outputTest <- data.frame(testData$ID,testPreds)
  colnames(outputTest) <- c("ID","prediction")
  return(outputTest)
}

predictValidationF1 <- function(mod){
  validatePreds <- predict(mod,validationData,type="response")
  confMatrixValidation <- caret::confusionMatrix(validatePreds,validationData$target,mode = 'prec_recall',positive="1")
  print(confMatrixValidation)
  return(confMatrixValidation$byClass[[7]])
  
}

## RandomForest 

rf_Model_Basic <- randomForest::randomForest(target~., trainData, keep.forest=TRUE, nTree = 100)
predictOnTrainAndValidation(rf_Model_Basic)
outputBasicTest <- predictOnTest(rf_Model_Basic)
table(outputBasicTest$target)
write.csv(outputBasicTest,"SampleSubmission_Basic.csv")

mTry <- randomForest::tuneRF(trainData[,-64], trainData$target,ntreeTry = 100, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot=TRUE)
print(mTry)

best.m <- mTry[mTry[, 2] == min(mTry[, 2]), 1]

#??randomForest::randomForest

set.seed(999)

rf_Mtry_Best <- randomForest::randomForest(target~.,data=trainData, mtry = 49, importance = TRUE, ntree = 50)
## with smote sampling, mTry is chosen as 22 - 25 
## with smote sampling , Mtry coming as 33
##rf_Mtry_Best <- randomForest::randomForest(target~.,data=trainData, mtry = 33, importance = TRUE, ntree = 50)

outputMtryTest <- predictOnTest(rf_Mtry_Best)
table(outputMtryTest$prediction)
write.csv(outputMtryTest,"SampleSubmission_Mtry.csv")


## lets try with different tree sizes

for(size in seq(50,200,10)){
  print(size)
  rf_Mtry_Best_temp <- randomForest::randomForest(target~.,data=trainData, mtry = 49, importance = TRUE, ntree = size)
  print(predictValidationF1(rf_Mtry_Best_temp))
  outputMtryTestTemp <- predictOnTest(rf_Mtry_Best_temp)
  print(table(outputMtryTestTemp$prediction))
}

## optimal number of trees was given at 90 , with respect to validation accuracy
rf_Mtry_Best_nTree <- randomForest::randomForest(target~.,data=trainData, mtry = 49, importance = TRUE, ntree = 90)
## smote sampling - 
##rf_Mtry_Best_nTree <- randomForest::randomForest(target~.,data=trainData, mtry = 33, importance = TRUE, ntree = 100)

outputMtryBestnTreeTest <- predictOnTest(rf_Mtry_Best_nTree)
table(outputMtryBestnTreeTest$prediction)
write.csv(outputMtryBestnTreeTest,"SampleSubmission_Mtry_nTree.csv")
#save(rf_Mtry_Best_nTree,file="randomForest_Mtry49_nTree90.rda")


## 53.12 %

mTry_nTree90 <- randomForest::tuneRF(trainData[,-64], trainData$target,ntreeTry = 150, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot=TRUE)
##mTry = 49, nTree= 90 is the best combination so far

## Can we drop any variables ??

randomForest::varImpPlot(rf_Mtry_Best_nTree)
rf_Imp_Attr = data.frame(rf_Mtry_Best_nTree$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]
summary(rf_Imp_Attr)

??randomForest::randomForest
## randomForest - nTree - 150, mTry - 49
rf_mTry_nTree150 = randomForest::randomForest(target~.,data=trainData, mtry = 49, ntree = 150)
f1validationtree150 <- predictValidationF1(rf_mTry_nTree150)

opmtryntree150 <- predictOnTest(rf_mTry_nTree150)
table(opmtryntree150$prediction)
write.csv(opmtryntree150,"SampleSubmission_Mtry_nTree150.csv")
save(rf_mTry_nTree150, file = "randomforest_mtry49_ntree150.rda")
## validation F1 score - 0.5613577, test F1 score - 55.07 %
mTry_nTree150_tune <- randomForest::tuneRF(trainData[,-64], trainData$target,ntreeTry = 150, stepFactor = 1.5, improve = 0.01, trace = TRUE, plot=TRUE)

predictValidationF1(rf_mTry_nTree150)
predictOnTrainAndValidation(rf_mTry_nTree150)

## what if i take only top 20 variables -- Lets do it later !!

## Boosting(ada, xgboost)

## smote , decision tree

## smote - tried with diff over sampling percents - not much accuracy shown


## Random Forest with C5 Imp attributes
traindata_c5imp

rf_basic_c5Imp <- randomForest::randomForest(target~.,data=traindata_c5imp,ntree=150)

predictValidationF1(rf_basic_c5Imp)
outputTestRfBasicImpC5 <- predictOnTest(rf_basic_c5Imp)
write.csv(outputTestRfBasicImpC5,"SampleSubmission_C5ImpRf.csv")
save(rf_basic_c5Imp,file="randomforest_c5imp_basic.rda")
summary(rf_basic_c5Imp)

## Tuning Imp Rf
mTry_nTree150_Imp_tune <- randomForest::tuneRF(traindata_c5imp[,-21], traindata_c5imp$target,ntreeTry = 150, stepFactor = 0.5, improve = 0.01, trace = TRUE, plot=TRUE)
best.m <- mTry_nTree150_Imp_tune[mTry_nTree150_Imp_tune[, 2] == min(mTry_nTree150_Imp_tune[, 2]), 1]

## best.m =16 , all are selected
rf_mTry16_nTree150 = randomForest::randomForest(target~.,data=traindata_c5imp, mtry = 16, ntree = 150)
predictValidationF1(rf_mTry16_nTree150)

## is there a better nTree??
for(size in c(90,170)){
  print(size)
  rf_Mtry_Best_temp <- randomForest::randomForest(target~.,data=traindata_c5imp, mtry = 16, importance = TRUE, ntree = size)
  print(predictValidationF1(rf_Mtry_Best_temp))
  outputMtryTestTemp <- predictOnTest(rf_Mtry_Best_temp)
  print(table(outputMtryTestTemp$prediction))
}

rf_mTry16_nTree170 = randomForest::randomForest(target~.,data=traindata_c5imp, mtry = 16, ntree = 170)
predictValidationF1(rf_mTry16_nTree170)
output_c5impRf_Tuned <- predictOnTest(rf_mTry16_nTree170)
write.csv(output_c5impRf_Tuned, "SampleSubmission_c5ImpRfTuned.csv")
save(rf_mTry16_nTree170,file="randomforest_c5imp_tuned.rda")



## Final - rf - mtry -16, nTree - 1000

rf_mTry16_nTree1000 = randomForest::randomForest(target~.,data=traindata_c5imp, mtry = 16, ntree = 1000)
predictValidationF1(rf_mTry16_nTree1000)
output_c5impRf_Tuned <- predictOnTest(rf_mTry16_nTree1000)
write.csv(output_c5impRf_Tuned, "SampleSubmission_c5ImpRfTuned_nTree1000.csv")

rf_mTry20_nTree1000 = randomForest::randomForest(target~.,data=traindata_c5imp, mtry = 20, ntree = 1000)
predictValidationF1(rf_mTry20_nTree1000)
output_c5impRf_Tuned <- predictOnTest(rf_mTry20_nTree1000)
write.csv(output_c5impRf_Tuned, "SampleSubmission_c5ImpRfTuned_mtry20_nTree1000.csv")






