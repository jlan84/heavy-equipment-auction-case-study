# <div align="center">_**Predicting Heavy Equipment Price**_</div>
## <div align="center">_**Team V: Justin, Alex W. and Chuck - Regression Case - July 31, 2020**_</div>

![header image](https://github.com/jlan84/regression-case-study/blob/master/predict_auction_price/images/construction.png)

## The Data:
The data is split into three parts:

**Train.csv** is the training set, which contains data through the end of 2011.
**Valid.csv** is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
**Test.csv** is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
The key fields are in train.csv are:

**SalesID:** the uniue identifier of the sale
**MachineID:** the unique identifier of a machine.  A machine can be sold multiple times
**saleprice:** what the machine sold for at auction (only provided in train.csv)
**saledate:** the date of the sale

There are several fields towards the end of the file on the different options a machine can have.  The descriptions all start with _"machine configuration"_ in the data dictionary.  Some product types do not have a particular option, so all the records for that option variable are null for that product type.  Also, some sources did not provide good option and/or hours data.
The _machine_appendix.csv_ file contains the correct year manufactured for a given machine along with the make, model, and product class details. There is one machine id for every machine in all the competition datasets (training, evaluation, etc.).

We will clean the data in two passes. The first will take advantage of intuitive selection of meaningful features. Secondly, we will immediately perform a scatter matrix to identify features that don't offer value based on a noticeable lack of correlation. 

## Features to Keep:
YearMade
Machine Hours
Saledate
Saleprice
ProductSize
State
ProductGroupDesc
Product 
DriveSystem
Enclosure
Forks
Pad_Type
Ride_Type
Transmission
Turbocharged
Tip_Control
Track_Type
Coupler
Differential Steer
Blade_Type
Thumb
Grouser_type
Blade_type
Differential_Type
Steering_Control
Backhoe_Mounting
Pattern_Changer
Blade_Width
Hydraulic_Flow


## Case Study Goal:
Predict the sale price of a particular piece of heavy equipment at auction based
on its usage, equipment type, and configuration.  The data is sourced from auction
result postings and includes information on usage and equipment configurations.

## Team Organization:
We approached this project from a divide and conquer perspective. Firstly, discussing the problem at hand, and the features that are given. Which needed to be included, cleaned and analyzed. Alex initially focused on the cleaning of the data and creating usable quantitative values for the data on an individual column level. While Justin attacked cleaning from additional directions, and keeping/removing columns of greater importance or non-value added features. Chuck focused directly on constructing the necessary functions for cross validation, various alpha applications and selecting optimal alpha. For both Ridge and Lasso.

Upon realizing clean and organized data, we all regathered to implement the functions. Analyzing results and selecting the best direction to move forward with predicting 2012 prices for equipment based on the provided features.

We all worked on the README report as individual assignments were completed - updating it based on the person with the greatest knowledge of the divided workload.

## Evaluation
The evaluation of our model has been based on Root Mean Squared Log Error.
Which is computed as follows:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values (predicted auction sale prices) 
and *a<sub>i</sub>* are the actual values (the actual auction sale prices).

## Data Cleaning and Preperation:
ENTER DATA CLEANING AND DETAILS HERE





## Overview of the score_model.py script
Included is a score function to test your predictions of the test set against the provided hold out test set.  This follows a common setup in competitions such as Kaggle, where this came from.  In these types of setups, there is a labeled train set to do your modeling and feature tuning.  There is also a provided hold-out test set to compare your predictions against.  You will need to fit a model on the training data and get a prediction for all the data in the test set.  You will then need to create csv containing the field 'SalesID' and 'SalePrice' (must match exactly).  This will be the input parameter to running the function.    
Example:
In terminal:
```
python score_model.py <path to csv file>
```


## Credit
This case study is based on [Kaggle's Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition.  The best RMSLE was only 0.23 (obviously lower is better).  Note
that if you were to simply guess the median auction price for all the pieces of equipment in
the test set you would get an RMSLE of about 0.7.
