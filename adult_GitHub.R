# Import Data 
dat <- read.csv("../input/adult-census-income/adult.csv", stringsAsFactors = T, header = T)
summary(dat)
str(dat)

# 1. remove unnecessary variables: fnlwgt, education, and relationship
dat = dat[,-c(3,4,8)] 
summary(dat)

# 2. Clean the Outliers by using IQR method
## cleaning outliers in age ##
summary(dat$age)
Q1_age = 28
Q3_age = 48
IQR_age = Q3_age - Q1_age#IQR = Q3 - Q1
IQR_age

# Find lowest value (LowerWhisker = Q1 - 1.5 * IQR_age) 
LowerW_age = Q1_age - (1.5*IQR_age)
LowerW_age

# Find upper value (UpperWhisker = Q3 + 1.5 * IQR_age)
UpperW_age = Q3_age + 1.5 * IQR_age
UpperW_age

# Find observations above 78 (as UpperW_age =78)
dat = subset(dat, age <= 78)

## Cleaning outliers in education.num ##
summary(dat$education.num)
Q1_education.num  = 9
Q3_education.num  = 12
IQR_education.num = Q3_education.num  - Q1_education.num
IQR_education.num 

# Find lowest value (LowerWhisker = Q1 - 1.5 * IQR_education.num) 
LowerW_education.num  = Q1_education.num - 1.5*IQR_education.num
LowerW_education.num 

# Find upper value: (UpperWhisker = Q3 + 1.5 * IQR_education.num)
UpperW_education.num  = Q3_education.num  + 1.5*IQR_education.num
UpperW_education.num 

# Find observations below 4.5
dat = subset(dat, education.num >= 4.5)

## Cleaning outliers in capital.gain ##
library(ggplot2)
summary(dat$capital.gain)
hist(dat$capital.gain)
box_plot = ggplot(dat, aes(x=capital.gain))+ geom_boxplot()
box_plot # 99999 would be a potential outlier
dat = subset(dat, capital.gain < 99999) 

# 3. Reclassifying Categorical Variables
## Change the "?" to Unknown ##
dat$occupation = gsub("?", "Unknown", dat$occupation, fixed = T )
dat$occupation =as.factor(dat$occupation)
summary(dat$occupation)

dat$workclass = gsub("?", "Unknown", dat$workclass, fixed = T )
dat$workclass =as.factor(dat$workclass)
summary(dat$workclass)

## Reclassify field values ##

## For workclass ##
# Grouping "Federal-gov" "Local-gov", and "State-gov" into "Gov"
levels(dat$workclass)
levels(dat$workclass)[c(1,2,7)] = 'Gov'
# Grouping "Self-emp-inc" and "Self-emp-not-inc" into "Self-emp"
levels(dat$workclass)
levels(dat$workclass)[4:5] = 'Self-emp'
levels(dat$workclass)

## For marital.status ##
levels(dat$marital.status)
levels(dat$marital.status)[c(2,3,4)] = 'Married'

## For native.country ##
t1 = table(dat$native.country) 
prop.table(t1) 
# Since 90% records are from the US, we group the variable native.country into "non-US" and "US"
levels(dat$native.country)
levels(dat$native.country)[c(28)] = 'United-States'
levels(dat$native.country)[c(1:27,29:41)] = 'Non-U.S.'

## For occupation ##
levels(dat$occupation)
levels(dat$occupation)[c(6,8,9)] = 'Service'
levels(dat$occupation)[c(4,8)] = 'Professional/Managerial'
levels(dat$occupation)[c(1,7)] = 'Administration'

# 4. Min-Max normalization
datnorm = dat
head(datnorm)
for (i in c(1, 3, 8, 9, 10)){
  mindf = min(datnorm[,i])
  maxdf = max(datnorm[,i])
  datnorm[,i] =(datnorm[,i] - mindf)/(maxdf - mindf)
}
head(datnorm)

# 5. Creating training and test data set
# Divide the dataset into 2 portions in the ratio of 75: 25 for the training and test data set respectively.
DF=datnorm
set.seed(123)
samp = sample(1:nrow(DF),round(0.75*nrow(DF)))
DF.training = DF[samp,]
DF.test= DF[-samp,]

# 6. Create dummy variable
library('caret')

## DF.training - create dummy variable ##
dmy.training = dummyVars(" ~ .", data = DF.training)
training.dmy = data.frame(predict(dmy.training, newdata = DF.training))

# Dummy variables are created. We have to remove the income<=50k column 
names(training.dmy) 
training.dmy = training.dmy[-37] # Remove column 37 which is income <=50
names(training.dmy)[names(training.dmy) == "income..50K"] = "income.more.50k" # Rename the income>50k column

## DF.test - create dummy variable ##
dmy.test = dummyVars(" ~ .", data = DF.test)
test.dmy = data.frame(predict(dmy.test, newdata = DF.test))

# Dummy variables are created. We have to remove the income<=50k column   
names(test.dmy) 
test.dmy = test.dmy[-37] # Remove column 41 which is income<=50
names(test.dmy)[names(test.dmy) == "income..50K"] = "income.more.50k" # Rename the income>50k column

# 7. Create a matrix for model comparison
Model.Com = data.frame(matrix(c(0), nrow=9, ncol=12))
colnames(Model.Com) = (c('Models','True Pos', 'True Neg','False Pos', 'False Neg','Overall Error Rate(%)','Proportion of FP(%)',
                         'Proportion of FN(%)', 'Accuracy(%)','Sensitivity(%)','Specificity(%)','Decision Cost($)'))
Model.Com$Models = (c('nnet10', 'nnet5', 'neuralnet3','knn20','knn10','knn5','CART Tree', 'C4.5Tree1','*C4.5Tree1.Cost'))
Model.Com

### 8a. Neural Networks by using nnet ###
library(nnet) # Using nnet() to create neural networks
nnet10 = nnet(income~., data=DF.training, size=10, maxit = 500) 
nnet5 = nnet(income~., data=DF.training, size=5, maxit = 500) 

# Prediction
estincome.nnet10 = predict(nnet10, DF.test, type = 'class')
estincome.nnet5 = predict(nnet5, DF.test, type = 'class')
estincome.nnet10 = as.factor(estincome.nnet10)
estincome.nnet5 = as.factor(estincome.nnet5)

# Confusion Matrix (using caret to create Confusion Matrix, income > $50,000 considered positive)
cm.nnet10 = confusionMatrix(reference = DF.test$income, data = estincome.nnet10, positive = '>50K')
cm.nnet5 = confusionMatrix(reference = DF.test$income, data = estincome.nnet5, positive = '>50K')
cm.nnet10$table
cm.nnet5$table

### 8b. Neural Networks by using neuralnet ###
library(neuralnet)
names(training.dmy)
neuralnet3 = neuralnet(income.more.50k~., data = training.dmy, hidden=3, linear.output = F, stepmax = 1e6)

# plot the neural network
plot(neuralnet3, show.weights=TRUE)
#check the weight
neuralnet3$weights

# Prediction
estincome.neuralnet3= predict(neuralnet3, test.dmy, method='class')

# Create a variable 'pred.income' and fill with '<=50k'
pred.neuralnet3 = rep('<=50K', length(estincome.neuralnet3))

# Using a threshold of 0.5 to determine if income is >50k
pred.neuralnet3 [estincome.neuralnet3>=0.5] = '>50K'
pred.neuralnet3 = as.factor(pred.neuralnet3)

# Confusion Matrix (using caret to create Confusion Matrix, income > $50,000 considered positive)
cm.neuralnet3 = confusionMatrix(reference = DF.test$income, data = pred.neuralnet3, positive = '>50K')
cm.neuralnet3$table

### 9. K-nearest Neighbor ###
library(class)
knn.training = training.dmy[-37] # Remove target variable
knn.test = test.dmy[-37]

# This code takes the income factor from the data frame and creates DF_train_labels and DF_test_labels.
DF_train_labels = DF$income[samp] # Real results
DF_test_labels = DF$income[-samp]

# Try different k values
estknn.20 = knn(knn.training, knn.test , DF_train_labels, k=20)
estknn.10 = knn(knn.training, knn.test , DF_train_labels, k=10)
estknn.5 = knn(knn.training, knn.test , DF_train_labels, k=5)

# Confusion Matrix (Income > $50,000 considered positive)
cm.knn20 = confusionMatrix(reference = DF.test$income, data = estknn.20, positive = '>50K')
cm.knn10 = confusionMatrix(reference = DF.test$income, data = estknn.10, positive = '>50K')
cm.knn5 = confusionMatrix(reference = DF.test$income, data = estknn.5, positive = '>50K')
cm.knn20$table
cm.knn10$table
cm.knn5$table

### 10. Decision Tree ###
library(rpart)
library(rpart.plot)

# 10a. CART Decision Tree
Ctree1 = rpart(income~., data=DF.training, method = "class", model = TRUE,control = rpart.control(minsplit = 1000))
rpart.plot(Ctree1) #Tree with Probabilities

# Prediction
realincome = DF.test$income
estincome.Ctree1 = predict(Ctree1, DF.test, type="class")

# Confusion Matrix (Income > $50,000 considered positive)
cm.Ctree1 = confusionMatrix(reference = DF.test$income, data = estincome.Ctree1, positive = '>50K')
cm.Ctree1$table

# 10b. C4.5 Decision Tree
library(C50)
# Define x & Y
x=DF.training[-12]
y=DF.training$income

C.50tree1 = C5.0(x,y, control = C5.0Control(minCases = 100))

# Prediction Table
est.C.50tree1 = predict(C.50tree1, DF.test, type='class')

# Confusion Matrix (Income > $50,000 considered positive)
cm.C.50tree1 = confusionMatrix(reference = DF.test$income, data = est.C.50tree1, positive = '>50K')
cm.C.50tree1$table

## 12. Misclassification cost adjustment ##
costm = matrix(c(1,2,1,1), byrow = FALSE, 2,2) #increase the false negative into 2

# Create the tree
C.50tree1.cost = C5.0(x,y, costs = costm,control = C5.0Control(minCases = 100))

# Prediction Table
est.C.50tree1.cost = predict(C.50tree1.cost, DF.test, type='class')

# Confusion Matrix (Income > $50,000 considered positive)
cm.C.50tree1.cost = confusionMatrix(reference = DF.test$income, data = est.C.50tree1.cost, positive = '>50K')
cm.C.50tree1.cost$table

## Decision Cost Analysis (Final result) - Putting the results to the Model Comparison Table
library(dplyr)
library(kableExtra)
library(formattable)

# Neural Networks
Model.Com[1,2] = nnet10.True.Positive = cm.nnet10$table[2,2]
Model.Com[1,3] = nnet10.True.Negative = cm.nnet10$table[1,1]
Model.Com[1,4] = nnet10.False.Positive = cm.nnet10$table[2,1]
Model.Com[1,5] = nnet10.False.Negative = cm.nnet10$table[1,2]

Model.Com[2,2] = nnet5.True.Positive = cm.nnet5$table[2,2]
Model.Com[2,3] = nnet5.True.Negative = cm.nnet5$table[1,1]
Model.Com[2,4] = nnet5.False.Positive = cm.nnet5$table[2,1]
Model.Com[2,5] = nnet5.False.Negative = cm.nnet5$table[1,2]

Model.Com[3,2] = neuralnet3.True.Positive = cm.neuralnet3$table[2,2]
Model.Com[3,3] = neuralnet3.True.Negative = cm.neuralnet3$table[1,1]
Model.Com[3,4] = neuralnet3.False.Positive = cm.neuralnet3$table[2,1]
Model.Com[3,5] = neuralnet3.False.Negative = cm.neuralnet3$table[1,2]

# Knn
Model.Com[4,2] = knn20.True.Positive = cm.knn20$table[2,2]
Model.Com[4,3] = knn20.True.Negative = cm.knn20$table[1,1]
Model.Com[4,4] = knn20.False.Positive = cm.knn20$table[2,1]
Model.Com[4,5] = knn20.False.Negative = cm.knn20$table[1,2]

Model.Com[5,2] = knn10.True.Positive = cm.knn10$table[2,2]
Model.Com[5,3] = knn10.True.Negative = cm.knn10$table[1,1]
Model.Com[5,4] = knn10.False.Positive = cm.knn10$table[2,1]
Model.Com[5,5] = knn10.False.Negative = cm.knn10$table[1,2]

Model.Com[6,2] = knn5.True.Positive = cm.knn5$table[2,2]
Model.Com[6,3] = knn5.True.Negative = cm.knn5$table[1,1]
Model.Com[6,4] = knn5.False.Positive = cm.knn5$table[2,1]
Model.Com[6,5] = knn5.False.Negative = cm.knn5$table[1,2]

# CART tree
Model.Com[7,2] = Ctree1.True.Positive = cm.Ctree1$table[2,2]
Model.Com[7,3] = Ctree1.True.Negative = cm.Ctree1$table[1,1]
Model.Com[7,4] = Ctree1.False.Positive = cm.Ctree1$table[2,1]
Model.Com[7,5] = Ctree1.False.Negative = cm.Ctree1$table[1,2]

# C4.5 trees
Model.Com[8,2] = C.50tree1.True.Positive = cm.C.50tree1$table[2,2]
Model.Com[8,3] = C.50tree1.True.Negative = cm.C.50tree1$table[1,1]
Model.Com[8,4] = C.50tree1.False.Positive = cm.C.50tree1$table[2,1]
Model.Com[8,5] = C.50tree1.False.Negative = cm.C.50tree1$table[1,2]

# C4.5 trees with cost adjustment
Model.Com[9,2] = C.50tree1.cost.True.Positive = cm.C.50tree1.cost$table[2,2]
Model.Com[9,3] = C.50tree1.cost.True.Negative = cm.C.50tree1.cost$table[1,1]
Model.Com[9,4] = C.50tree1.cost.False.Positive = cm.C.50tree1.cost$table[2,1]
Model.Com[9,5] = C.50tree1.cost.False.Negative = cm.C.50tree1.cost$table[1,2]

# Calculate the formula
Overall.Error.Rate = (Model.Com[,4] + Model.Com[,5])/(Model.Com[,2] + Model.Com[,3]+ Model.Com[,4] + Model.Com[,5])*100
PFP = ((Model.Com[,4])/(Model.Com[,4] + Model.Com[,2]))*100
PFN = (Model.Com[,5]/(Model.Com[,5] + Model.Com[,3]))*100
Sensitivity = Model.Com[,2]/(Model.Com[,2] + Model.Com[,5])*100 #TP/(TP + FN)
Specificity = Model.Com[,3]/(Model.Com[,4] + Model.Com[,3])*100  #TN/(FP + TN)
Decision.Cost = (Model.Com[,3]*0 )+ (Model.Com[,2]*(-200)) +(Model.Com[,5]*0) +(Model.Com[,4]*500)

# Calculate the results
Model.Com[1:9,6]=round(Overall.Error.Rate, digits = 3)
Model.Com[1:9,7]=round(PFP, digits = 3)
Model.Com[1:9,8]=round(PFN, digits = 3)
Model.Com[1:9,10]=round(Sensitivity, digits = 3)
Model.Com[1:9,11]=round(Specificity, digits = 3)
Model.Com[1:9,12]=Decision.Cost

# Calculate the overall Accuracy again
Overall.Accuracy = round(100-Model.Com[,6], digits = 3)
Model.Com[1:9,9]=Overall.Accuracy

#kable extra
Model.Com %>%
  mutate(
    Models = cell_spec(Models, color = "white", bold = T, background = spec_color(1:9, end = 0.9, option = "D", direction = 1)),
    `Decision Cost($)` = cell_spec(`Decision Cost($)`, "html", color = ifelse(`Decision Cost($)` > 0, "Red", "Green"), bold = T),
    `Overall Error Rate(%)` = color_tile("Gold", "LightYellow")(`Overall Error Rate(%)`),
    `Proportion of FP(%)` = color_tile("Gold", "LightYellow")(`Proportion of FP(%)`),
    `Proportion of FN(%)` = color_tile("Gold", "LightYellow")(`Proportion of FN(%)`),
    `Accuracy(%)` = color_tile("Linen", "pink")(`Accuracy(%)`),
    `Sensitivity(%)` = color_tile("Linen", "pink")(`Sensitivity(%)`),
    `Specificity(%)` = color_tile("Linen", "pink")(`Specificity(%)`) 
  )%>%
  kable(escape = F, align = "l") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed","responsive"))%>%
  row_spec(0, font_size = 9)%>%
  column_spec(2:11, bold = T, color = "DimGray")%>%
  footnote(general = "For the decision cost, negative costs represent estimated profits.")