# machinelearningassign

library("plyr")

library("data.table")

library("reshape2")

file.path <- "/Users/emmasun/Desktop/data science/cleaning data"

setwd(file.path)

##read the training data into R##

require(data.table)

setInternet2(TRUE)

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

D <- fread(url)

##read the testing data set##

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

DTest <- fread(url)

##see if theres any NA##

isAnyMissing <- sapply(DTest, function (x) any(is.na(x) | x == ""))

##set predictors##

meanPredictor <- !isAnyMissing & grepl("belt|[^(fore)]arm|dumbbell|forearm", names(isAnyMissing))

predCandidates <- names(isAnyMissing)[isPredictor]

predCandidates

##adjust the dataset for prediction 

varToInclude <- c("classe", predCandidates)

D <- D[, varToInclude, with=FALSE]

dim(D)

names(D)

D <- D[, classe := factor(D[, classe])]

D[, .N, classe]

##set the training set and the testing set##

require(caret)

seed <- as.numeric(as.Date("2014-10-26"))

set.seed(seed)

inTrain <- createDataPartition(D$classe, p=0.6)

DTrain <- D[inTrain[[1]]]

DProbe <- D[-inTrain[[1]]]

##centering and scaling##

X <- DTrain[, predCandidates, with=FALSE]

prePro <- preProcess(X)

prePro

XCD <- predict(preProc, X)

DTrainCD <- data.table(data.frame(classe = DTrain[, classe], XCD))

##apply the rules##

X <- DProbe[, predCandidates, with=FALSE]

XCD <- predict(preProc, X)

DProbeCD <- data.table(data.frame(classe = DProbe[, classe], XCD))

##check for variance##

nzv <- nearZeroVar(DTrainCD, saveMetrics=TRUE)

if (any(nzv$nzv)) nzv else message("No variables with near zero variance")

hitGroup <- function (data, regex) {
  
  col <- grep(regex, names(data))
  
  col <- c(col, which(names(data) == "classe"))
  
  require(reshape2)
  
  n <- nrow(data)
  
  DMelted <- melt(data[, col, with=FALSE][, rownum := seq(1, n)], id.vars=c("rownum", "classe"))
  
  require(ggplot2)
  
  ggplot(DMelted, aes(x=classe, y=value)) +
    
    geom_violin(aes(color=classe, fill=classe), alpha=1/2) +
    
    facet_wrap(~ variable, scale="free_y") +
    
    scale_color_brewer(palette="Spectral") +
    
    scale_fill_brewer(palette="Spectral") +
    
    labs(x="", y="") +
    
    theme(legend.position="none")
}

hitGroup(DTrainCD, "belt")

hitGroup(DTrainCD, "[^(fore)]arm")

hitGroup(DTrainCD, "dumbbell")

hitGroup(DTrainCD, "forearm")

##Train the model##

require(parallel)

require(doParallel)

cl <- makeCluster(detectCores() - 1)

registerDoParallel(cl)

##set the parameters##

ctrl <- trainControl(classProbs=TRUE,

                     savePredictions=TRUE,

                     allowParallel=TRUE)

method <- "rf"

system.time(trainingModel <- train(classe ~ ., data=DTrainCD, method=method))

stopCluster(c1)

##Evaluate the model##

trainingModel

hat <- predict(trainingModel, DTrainCD)

confusionMatrix(hat, DTrain[, classe])

hat <- predict(trainingModel, DProbeCD)

confusionMatrix(hat, DProbeCD[, classe])

##Display##

varImp(trainingModel)

trainingModel$finalModel

save(trainingModel, file="trainingModel.Data")

####################################################Prediction####################################################

load(file="trainingModel.RData", verbose=TRUE)

DTestCD <- predict(preProc, DTest[, predCandidates, with=FALSE])

hat <- predict(trainingModel, DTestCD)

DTest <- cbind(hat , DTest)

subset(DTest, select=names(DTest)[grep("belt|[^(fore)]arm|dumbbell|forearm", names(DTest), invert=TRUE)])



