
## What are the assumptions of linear regression?
Linear regression is used to understand the relation between features (X) and target (y). Before we train the model, we need to meet a few assumptions:

The residuals are independent 
There is a linear relation between X independent variable and y dependent variable. 
Constant residual variance at every level of X
The residuals are normally distributed. 

Note: the residuals in linear regression are the difference between actual and predicted values. 

## What is Ensemble learning?
Ensemble learning is used to combine the insights of multiple machine learning models to improve the accuracy and performance metrics. 

Simple ensemble methods:

Mean/average: we average the predictions from multiple high-performing models.
Weighted average: we assign different weights to machine learning models based on the performance and then combine them.  
Advance ensemble methods:

Bagging is used to minimize variance errors. It randomly creates the subset of training data and trains it on the models. The combination of models reduces the variance and makes it more reliable compared to a single model. 
Boosting is used to reduce bias errors and produce superior predictive models. It is an iterative ensemble technique that adjusts the weights based on the last classification. Boosting algorithms give more weight to observations that the previous model predicted inaccurately.

![bagging v. boosting](https://images.datacamp.com/image/upload/v1662734968/15_Bagging_and_Boosting_dfc2b5ae07.jpg)
[https://www.datacamp.com/blog/top-machine-learning-interview-questions]
