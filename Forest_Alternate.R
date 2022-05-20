# Fireforest we need to implement Support Vector regression as the output variable is continuous
#load the dataset

FF = fireforests

str(FF)

install.packages("tidyverse")
install.packages("magrittr")
library(tidyverse)
library(magrittr)
library(dplyr)

FF <- FF %>% mutate_if(is.factor,as.numeric)
View(FF)

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the glass data

FF_norm <- as.data.frame(lapply(FF[1 : 30], normalize))
View(FF_norm)

#install package e1071 for SVR

install.packages("e1071")
library(e1071)

FFn_train = FF_norm[1:400,]
str(FFn_train)

library("car")
FFn_test = FF_norm[401 : 517,]

svm_model <- svm(area ~., data = FFn_train, scale = F, kernel = 'linear')
pred <- predict(svm_model,FFn_test)
actual <- FFn_test$area
error <- actual - pred


test.rmse <- sqrt(mean(error**2))
test.rmse #0.1003

train.rmse <- sqrt(mean(svm_model$residuals**2))
train.rmse #0.099

svm_model$coefs # coefficients of the Model
svm_model$rho   # rho i.e constant value -0.0988

# Tuned SVM Model 
#Tune the SVM model
OptModelsvm=tune(svm, area ~., data = FF_norm ,ranges=list(elsilon=seq(0,1,0.1), cost=1:100, scale = F))

# Find the best model

BstModel=OptModelsvm$best.model

#Predict area using best model on train data
PredYBst=predict(BstModel,FFn_train)


#Install Package
install.packages("hydroGOF")

#Load Library
library(hydroGOF)

#Calculate RMSE of the best model 
RMSEBst=rmse(PredYBst,FFn_train$area) #train rmse is 0.08 which is still reduced

#Predict area using best model on test data

PredYBst1=predict(BstModel,FFn_test)

RMSE_test = rmse(PredYBst1,FFn_test$area) #0.092 

#By tuning the parameters we can achieve best right fit for test and train rmse


