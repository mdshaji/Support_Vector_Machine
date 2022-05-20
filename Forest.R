#####Support Vector Machines for classification model

# Analysizing the input and output variables
# Output Variable(y) - 
# Input Variable (x) - Other Factors

# Load the Dataset
Data <- read.csv(file.choose())

# Removing unnecessary columns
Data <- Data[3:30]
View(Data)
str(Data)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

summary(Data)
attach(Data)

# Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

# Boxplot Representation

boxplot(FFMC, col = "dodgerblue4",main = "FFMC")
boxplot(DMC, col = "dodgerblue4",main = "DMC")
boxplot(DC, col = "dodgerblue4",main = "DC")
boxplot(temp, col = "red", horizontal = T,main = "Temp")

# Histogram Representation

hist(FFMC,col = "orange", main = "FFMC" )
hist(DMC,col = "orange", main = "DMC")
hist(DC,col = "orange", main = "DC")
hist(temp,col = "red", main = "Temp")


# Or make a combined plot
#pairs(Start_up)   #  doesnt work as there is a categorical variable
#Scatter plot for all pairs of variables
plot(Data)

# correlation matrix
cor(Data)

#custom normalization function

normalise <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

Data$temp <- normalise(Data$temp)
Data$rain <- normalise(Data$rain)
Data$RH <- normalise(Data$RH)
Data$wind <- normalise(Data$wind)

# Converting area into Categorical
sum(Data$area < 5)
sum(Data$area >= 5)
Data$size <- NULL
Data$size <- factor(ifelse(Data$area < 5, 1, 0),
                      labels = c("small", "large"))

# Training a model on the data
train <- sample(x = nrow(Data), size = 400, replace = FALSE)

# Building the model using Polynomial Kernel
m.poly <- ksvm(size ~ temp + RH + wind + rain,
               data = Data[train, ],
               kernel = "polydot", C = 1)
m.poly
# Training Error = 0.3

# Building the model using Rbfdot Kernel
m.rad <- ksvm(size ~ temp + RH + wind + rain,
              data = Data[train, ],
              kernel = "rbfdot", C = 1)
m.rad
#Training error = 0.29

# Building the model using tanhdot Kernel
m.tan <- ksvm(size ~ temp + RH + wind + rain,
              data = Data[train, ],
              kernel = "tanhdot", C = 1)
m.tan
#Training error = 0.4775

# From the above 3 models m.rad(rbfdot) model gives less training error
# Now the model is tested on the test data
# Test Data Prediction

pred <- predict(m.rad, newdata = Data[-train, ], type = "response")
table(pred, Data[-train, "size"])
library(caret)
confusionMatrix(table(pred, Data[-train, "size"]), positive = "small")

# Accuracy of the model is 0.74
