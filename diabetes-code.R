# "Diabetes Diagnosis Project" by Daping Zhang (July 26, 2022)

# This project is to build a machine learning model to accurately predict whether
# or not the patients in the dataset have diabetes or not? 


# importing the libraries
options(digits = 3)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(kassambara/factoextra)) install.packages("kassambara/factoextra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "https://cran.rstudio.com/bin/macosx/contrib/4.2/randomForest_4.7-1.1.tgz")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.package("gam", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)        # also for ConfusionMatrix()
library(data.table)
library(ggplot2)
library(matrixStats)  # for colSds()
library(corrplot)
library(factoextra)   # for fviz_eig()
library(randomForest)
library(GGally)
library(gam)

# Pima Indians Diabetes Database:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/download/
  
# import the dataset
urlfile = "https://raw.githubusercontent.com/dapingz/Diabetes-Project/main/diabetes.csv"

# loading the dataset as data frame
diab_dat <- as.data.frame(read_csv(url(urlfile)))

# The dataset consists of several medical predictor variables include the number
# of pregnancies the patient has had, their BMI, insulin level, age, and so on, 
# and one target variable "Outcome".

# display the first 10 rows of the dataset in the table
knitr::kable(head(diab_dat, 10))

# see more tails about the  dataset
str(diab_dat)

# look at how many people have diabetes and how many do not
table(diab_dat$Outcome)

# look at the proportion of "no" and "yes"
prop.table(table(diab_dat$Outcome))


# Data Pre-processing & Data Explorations

# convert the Outcome to factor, while changing '1' and '0' to 'yes' and 'no' to be more meaningful. 
diab_dat$Outcome <- as.factor(ifelse(diab_dat$Outcome == "1", "Yes", "No"))

# Age Analysis and grouping ages
ggplot(aes(x = Age), data = diab_dat) +
  geom_histogram(binwidth = 2 , color = "white", fill = "#008080") +
  scale_x_continuous(limits = c(20,85), breaks = seq(20,85,5))

# divide Age into different age groups, create a temp dataset "diab_dat2"
diab_dat2 <- diab_dat

# create Age group column from "Yes" and "No"
diab_dat2$Age_group <- ifelse(diab_dat2$Age < 21, "<21", 
                         ifelse((diab_dat2$Age>=21) & (diab_dat2$Age<=25), "21-25", 
                           ifelse((diab_dat2$Age>25) & (diab_dat2$Age<=30), "26-30",
                             ifelse((diab_dat2$Age>30) & (diab_dat2$Age<=35), "31-35",
                               ifelse((diab_dat2$Age>35) & (diab_dat2$Age<=40), "36-40",
                                ifelse((diab_dat2$Age>40) & (diab_dat2$Age<=45), "41-45",
                                  ifelse((diab_dat2$Age>45) & (diab_dat2$Age<=50), "46-50",
                                   ifelse((diab_dat2$Age>50) & (diab_dat2$Age<=55), "51-55",
                                    ifelse((diab_dat2$Age>55) & (diab_dat2$Age<=60), "56-60",
                                      ">60")))))))))

# convert Age_group to factor
diab_dat2$Age_group <- factor(diab_dat2$Age_group, 
  levels=c('<21','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','>60'))

# barplot by Age_group, to see which age group is the most in the dataset
ggplot(aes(x = Age_group), data = diab_dat2) + geom_bar(fill='#008080')

# make a plot to find out which age group has the highest rate of diabetes
ggplot(diab_dat2, aes(x = Age_group, fill = Outcome)) + geom_bar(position = "fill") +
  scale_fill_manual(values = c("No"="#48C980", "Yes"="red"))


# Checking missing values

# first check if any null values in the dataset
sum(is.na(diab_dat))

# then check if any zero values
knitr::kable(head(diab_dat))

# a value of zero does not make sense and thus indicates missing value
# replace these zeros with the median of each column

# step1: replace all zeros with NAs
diab_dat[diab_dat == 0] <- NA

# step2: replace all "NA"s with the median of each column
diab_dat <- diab_dat %>% 
  mutate_if(is.numeric, function(x) ifelse(is.na(x), median(x, na.rm = T), x))

# check dataset again to see that all columns containing "0" are now replaced with median
knitr::kable(head(diab_dat))

# Splitting the dataset into independent and dependent datasets

# use row 1 to 8 as independent dataset and covert it to a matrix 
diab_x <- as.matrix(diab_dat[,1:8]) 

# use the Outcome as dependent dataset
diab_y <- diab_dat$Outcome 

# there are 768 samples and 8 predictors in the dataset (features matrix).
dim(diab_x)[1] 
dim(diab_x)[2] 

# there are 768 outcomes in the dependent dataset (target variable)
length(diab_y) 

# summary dependent dataset
summary(diab_y)

# check the proportions of No and Yes samples in the dataset

# see the proportion of No and Yes in a table.
prop.table(table(diab_y))

# see the proportion of No and Yes on the bar plot
data.frame(diab_y) %>% 
  ggplot(aes(diab_y)) +
  geom_bar(fill='#008080')

# Feature Scaling and Splitting Datasets 

# Scaling all predictor variables
x_centered <- sweep(diab_x, 2, colMeans(diab_x))
x_scaled <- sweep(x_centered, 2, colSds(diab_x), FUN = "/")

# After scaling, the standard deviation is 1 for all columns
colSds(x_scaled)
summary(x_scaled)

# Correlation Analysis

# corrplot
corr_mat <- cor(diab_x)
corrplot(corr_mat)

# plot correlation matrix
ggcorr(diab_x, label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE) +
  theme(legend.position = "none")


# Splitting dataset into training and test sets

# split the dataset into a training set (80%) and a test set (20%)
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(diab_y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- diab_y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- diab_y[-test_index]

# check the proportion in training set
prop.table(table(train_y))

# check the proportion in test set
prop.table(table(test_y))


# Building the Models

# Model 1: Logistic Regression Model

# train a glm model on the training set
train_glm <- train(train_x, train_y, method = "glm")

# generate predictions on the test set
glm_preds <- predict(train_glm, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_glm <- confusionMatrix(data = glm_preds, reference = test_y)
result_glm <- c(cm_glm$byClass[c("Sensitivity","Specificity",
                                 "Prevalence")],cm_glm$overall["Accuracy"])
result_glm

# Model 2: rpart Model

# train a rpart model on the training set
train_rpart <- train(train_x, train_y, method = "rpart")

# generate predictions on the test set
rpart_preds <- predict(train_rpart, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_rpart <- confusionMatrix(data = rpart_preds, reference = test_y)
result_rpart <- c(cm_rpart$byClass[c("Sensitivity","Specificity", 
                                     "Prevalence")], cm_rpart$overall["Accuracy"])
result_rpart

# Model 3: LDA Model

# train a LDA model on the training set
train_lda <- train(train_x, train_y, method = "lda")

# generate predictions on the test set
lda_preds <- predict(train_lda, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_lda <- confusionMatrix(data = lda_preds, reference = test_y)
result_lda <- c(cm_lda$byClass[c("Sensitivity","Specificity", 
                                 "Prevalence")], cm_lda$overall["Accuracy"])
result_lda

# Model 4: QDA Model

# train a QDA model on the training set
train_qda <- train(train_x, train_y, method = "qda")

# generate predictions on the test set
qda_preds <- predict(train_qda, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_qda <- confusionMatrix(data = qda_preds, reference = test_y)
result_qda <- c(cm_qda$byClass[c("Sensitivity","Specificity",
                                 "Prevalence")], cm_qda$overall["Accuracy"])
result_qda

# Model 5: Loess Model

# set.seed(5)
set.seed(5, sample.kind = "Rounding") # simulate R 3.5

# train a loess model on the training set
train_loess <- train(train_x, train_y, method = "gamLoess")

# generate predictions on the test set
loess_preds <- predict(train_loess, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_loess <- confusionMatrix(data = loess_preds, reference = test_y)

result_loess <- c(cm_loess$byClass[c("Sensitivity","Specificity", 
                                     "Prevalence")], cm_loess$overall["Accuracy"])
result_loess

# Model 6: K-Nearest Neighbours Model

# train a KNN model on the training set, and find the best value in the model
set.seed(7,sample.kind = "Rounding")
train_knn <- train(train_x, train_y, 
                   method = "knn",
                   tuneGrid = data.frame(k = seq(3, 21, 2)))

# determine the value of k
train_knn$bestTune # the best value is 10 or 21

# use the final value in the model to generate predictions on the test set
knn_preds <- predict(train_knn, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_knn <- confusionMatrix(data = knn_preds, reference = test_y)
result_knn <- c(cm_knn$byClass[c("Sensitivity","Specificity", 
                                 "Prevalence")], cm_knn$overall["Accuracy"])
result_knn

# Model 7: Random Forest model

set.seed(9,sample.kind = "Rounding")

# train a random forest model on the training set
train_rf <- train(train_x, train_y, 
                  method = "rf",
                  importance = TRUE,
                  tuneGrid = data.frame(mtry = c(3,5,7,9)))

# find the value of mtry gives the highest accuracy 
train_rf$bestTune  # the best value of mtry is 3 

# find the most important variable in the random forest model
varImp(train_rf) 

# use the best value of mtry in the model to generate predictions on the test set
rf_preds <- predict(train_rf, test_x)

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_rf <- confusionMatrix(data = rf_preds, reference = test_y)
result_rf <- c(cm_rf$byClass[c("Sensitivity","Specificity", 
                               "Prevalence")], cm_rf$overall["Accuracy"])
result_rf

# Model 8: Ensemble Model

# creating an ensemble model that combine all the models trained before
ensemble <- cbind(glm = glm_preds=="No", lda = lda_preds=="No", 
                  qda = qda_preds=="No", loess = loess_preds=="No", 
                  rf = rf_preds=="No", knn = knn_preds=="No", 
                  rpart = rpart_preds=="No")

# generate predictions
ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "No", "Yes")

# calculate the accuracy, sensitivity, specificity, and prevalence of the model on test set
cm_ensemble <- confusionMatrix(data = factor(ensemble_preds), reference = test_y)
result_ensmble <- c(cm_ensemble$byClass[c("Sensitivity","Specificity", "Prevalence")],
                    cm_ensemble$overall["Accuracy"])
result_ensmble

# create an ensemble list that shows the accuracy of these models
model_names <- c("Logistic regression","rpart","LDA","QDA","Loess",
                 "K nearest neighbors","Random forest","Ensemble")

models_accuracy <- c(cm_glm$overall["Accuracy"], cm_rpart$overall["Accuracy"], 
                     cm_lda$overall["Accuracy"], cm_qda$overall["Accuracy"],
                     cm_loess$overall["Accuracy"], cm_knn$overall["Accuracy"], 
                     cm_rf$overall["Accuracy"], cm_ensemble$overall["Accuracy"]) 
data.frame(Model = model_names, Accuracy = models_accuracy)

# select a best prediction model according to highest accuracy
best_model <- subset(models_accuracy, models_accuracy==max(models_accuracy))
best_model

# Model comparisons

# compare the accuracy of the models, and their performs in terms of sensitivity, specificity
result_all <- data.frame(rbind(result_glm,result_rpart,result_lda,result_qda,
                               result_loess,result_knn, result_rf,result_ensmble))
names(result_all) <- c("Sensitivity","Specificity", "Prevalence", "Accuracy")
result_all

#References

# 1. HarvardX PH125.8x Data Science: Machine Learning: 7.1 Final Assessment: Breast Cancer Prediction Project 
# 2. Kaggle: Pima Indians Diabetes Database, Dataset Notebooks
# 3. Kaggle: MIRI CHO,[Classification] Diabetes or Not (with basic 12ML)
# 4. Kaggle: LEARNING-MONK, PIMA Indian Diabetes - Logistic Regression with R
# 5. ggcorr: https://briatte.github.io/ggcorr

