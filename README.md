# Sparkify User Churn
1. [Installation](#installation)
2. [Objective](#objective)
3. [Results on a Small Dataset](#results-on-a-small-dataset)
4. [Results on a Medium Dataset](#results-on-a-medium-dataset)
5. [File Descriptions](#file-descriptions)
6. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Installation
The code was developed using Python 3.6.3. Necessary packages beyond the Python Standard Library are:

**For sparkify-user-churn-small.ipynb**
- matplotlib==3.3.4
- numpy==1.19.5
- pandas==0.23.3
- pyspark==2.4.3
- seaborn==0.11.1 (Sidenote for Udacity workspace: Please upgrade seaborn using `pip install seaborn --upgrade` because the default workspace contains an older version which does not support the function histplot)

The environment can be installed using [requirements.txt](https://github.com/MareikeHeller/sparkify-customer-churn/blob/main/requirements.txt).

**For sparkify-user-churn-medium.ipynb** 
- matplotlib==3.2.2
- numpy==1.18.5
- pandas==0.25.3
- pyspark==2.4.3
- seaborn==0.10.1

The environment can be installed using [requirements_medium.txt](https://github.com/MareikeHeller/sparkify-customer-churn/blob/main/requirements_medium.txt).

## Objective
This project tackles the **prediction of user churn** after data wrangling, exploratory data analysis, feature engineering as well as modeling and tuning different machine learning classification algorithms. The broader context involves the **cluster-computing framework Spark** in order to expand the methods to **big data** using the Spark ML DataFrame-based API.

The project uses datasets from a simulated audio streaming service provider **Sparkify** containing events from user interaction with their product. A small subset of events is used in this notebook to develop the methods prior to deployment in the cloud.

**Problem Statement:**

Users of the audio streaming service Sparkify can manage their involvement by upgrading from a free to paid version, downgrading vice versa or canceling the service altogether. The problem investigated in this project will be to predict user churn: the identification of users who have a higher probability of canceling the service in the near future. From a business perspective, a clean identification of user churn enables insights into the driving factors of adverting from the service. Furthermore, it motivates countermeasures targeted to a specific user group, such as providing incentives for staying involved.

This problem is solved by tuning and contrasting different machine learning classification algorithms that should distinguish churn users from non-churn users to find the most suited classification model.

**Metrics for Model Performance:**

Measurement of model performance is done by overall model **F1 score, accuracy and single feature importance**. The focus for the overall model performance should be on the F1 score as a weighted average of Precision and Recall1 instead of accuracy in this scenario because the label "churn" is imbalanced in the dataset (23% churn). Single feature importance gives insight into strong and weak predictors of churn leading to a more precise understanding of user churn in the first place and triggering improved feature engineering for refining the model in the future.

## Results on a Small Dataset
User churn is defined according to page type "Cancellation Confirmation". As could be shown throughout the analysis, this event occurs once per user and churn users do not return. **23% of users churn**. This yields a **perfectly consistent base for building a churn prediction model**.

The best model for churn prediction before hyperparameter tuning is the **Logistic Regression with the highest F1 score of 88%** which could not be persisted after hyperparameter tuning. The best model for churn prediction after hyperparameter tuning is the **Gradient-Boosted Tree Classifier with the highest F1 score of 86%**. Nevertheless, it is the worst performing model in terms of training time which could make it **harder to scale with larger datasets**.

Especially, the features **days since registration, page about, page roll advert and page thumbs down** stood out with highest impact on the prediction model. The exploratory data analysis showed less days since registration for churn users in comparison to non-churn users, indicating that **the decision to churn takes place in the first months after registration**. Users, who are already involved for a longer time period, are more likely to remain using the service. So, the business should focus on fresh users to prevent churn. Not surprisingly, interactions such as seeing advertising and giving negative feedback (thumbs down) are important features when predicting churn.

The implementation of machine learning **pipelines including scaling and cross-validation reduce the risk of data leakage** at different points in the process.

**Particularities and improvements** are discussed in the notebook.

## Results on a Medium Dataset
I lifted the same analyses and modelings on a larger medium-sized dataset to the cloud in IBM Watson Studio. The more data can be processed to create a prediction model, the more stable the final result will be. Spurious effects due to noise will be more likely to be canceled out and sensitivity to detect true effects will be enhanced. 

In fact, using the medium-sized data subset, the **Logistic Regression was the best prediction model before (F1 score = 96%) and after (F1 score = 82%) hyperparameter testing.** The **feature days since registration** consistently emerged as the most important feature again indicating that the decision to churn takes place in the first months after registration. This was already detected using the small data subset in the first place.

## File Descriptions
**requirements.txt**
- can be used to install the python environment for Udacity workspace (small data subset)

**requirements_medium.txt**
- can be used to install the python environment for cloud deployment on IBM (medium data subset)

**sparkify-user-churn-small.ipynb**
- using a small data subset in Udacity workspace
- notebook contains
  - data wrangling
  - exploratory data analysis
  - feature engineering
  - modeling and tuning different machine learning classification pipelines

**sparkify-user-churn-medium.ipynb**
- using a medium data subset in the cloud (IBM Watson Studio)
- notebook contains
  - data wrangling
  - exploratory data analysis
  - feature engineering
  - modeling and tuning different machine learning classification pipelines

Please note that the corresponding small and medium data files were provided by [Udacity](https://www.udacity.com/) and are *not* contained in this notebook due to large file sizes.

## Licensing, Authors, Acknowledgements
This is the capstone project related to the [Udacity Data Science Nanodegree](https://www.udacity.com/school-of-data-science). The data used in this project was kindly provided by [Udacity](https://www.udacity.com/).

1 | https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/

2 | https://mungingdata.com/pyspark/udf-dict-broadcast/ 

3 | https://medium.com/swlh/logistic-regression-with-pyspark-60295d41221

4 | https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c
