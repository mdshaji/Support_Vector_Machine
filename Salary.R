#####Support Vector Machines for classification model

# Analysizing the input and output variables
# Output Variable(y) - Salary
# Input Variable (x) - Other Factors

# Load the Dataset

# Training Data - Data file is imported by Text(base) to convert strings into factors
Salary_train <- read.csv(file.choose())
View(Salary_train)
str(Salary_train)
attach(Salary_train)

# Checking NA values
sum(is.na(Salary_train))
# There are no NA values available in dataset

# Test Data - Data file is imported by Text(base) to convert strings into factors
Salary_test <- read.csv(file.choose())
View(Salary_test)
str(Salary_test)
attach(Salary_test)

# Checking NA values
sum(is.na(Salary_test))
# There are no NA values available in dataset
       
# Exploratory Data Analysis

summary(Salary_train)
summary(Salary_test)

# Graphical Visualization 
library(ggplot2)

# Plots

plot(occupation,Salary, main = "Occupation")
plot(education,Salary, main = "Education")
plot(workclass,Salary, main = "Workclass")
plot(relationship,Salary, main = "Relationship")

# GG plots
ggplot(data= Salary_train,aes(x=Salary, y = age, fill = Salary)) +
  geom_boxplot() + ggtitle("Box Plot")

ggplot(data=Salary_train,aes(x=Salary, y = hoursperweek, fill = Salary)) +
  geom_boxplot() +
  ggtitle("Box Plot")

#Density Plot 

ggplot(data=Salary_train,aes(x = age, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = workclass, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = education, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')

ggplot(data=Salary_train,aes(x = educationno, fill = Salary)) +
  geom_density(alpha = 0.9, color = 'Violet')


# Training a model on the data ----
# Begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

Salary_classifier <- ksvm(Salary ~ ., data = Salary_train, kernel = "vanilladot")
?ksvm

## Evaluating model performance ----
# predictions on testing dataset
Salary_predictions <- predict(Salary_classifier, Salary_test)

table(Salary_predictions, Salary_test$Salary)
agreement <- Salary_predictions == Salary_test$Salary
table(agreement)
prop.table(table(agreement))

# FALSE = 0.1537185       
# TRUE =  0.8462815 

## Improving model performance ----
Salary_classifier_rbf <- ksvm(Salary ~ ., data = Salary_train, kernel = "rbfdot")
Salary_predictions_rbf <- predict(Salary_classifier_rbf, Salary_test)
agreement_rbf <- Salary_predictions_rbf == Salary_test$Salary
table(agreement_rbf)
prop.table(table(agreement_rbf))

# FALSE = 0.1455511     
# TRUE =  0.8544489
