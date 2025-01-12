## An investigation into Multiclass Classifier performance on Dry Beans dataset

### :pushpin: Project Overview

___

Publicly available Dry Bean dataset (Koklu, 2020), used in this experiment, was created from images of 13,611 grains of 7 basic dry bean varieties captured by a specific computer vision system (Koklu and Ozkan, 2020). These images were further processed to extract 16 features - 12 dimensions and 4 shape forms.

Five classical supervised machine learning algorithms (Support Vector Machine (SVM), Decision Tree (DT), k-Nearest Neighbors (KNN), Random Forest (RF) and Multilayer Perceptron (MLP)) were employed to classify beans. Subsequently, the three models with the highest average weighted F1-scores were chosen.

I then tuned the hyperparameters of these models to identify the one that achieves the best averaged weighted F1 score, optimizing it for this specific multi-class classification task.

### :pushpin: Table of Contents
___


### :pushpin: Data Sources
___

Publicly available [Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu, 2020) can be downloaded in *.xlsx* or *.arff* formats from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu).

### :pushpin: Development Environment and Dependencies
___

All scripts are implemented in Python.</br>
Jupyter notebook used for interactive data analysis and visual explorations. 

The [environment.yml] file lists all necessary Python packages and can be used to recreate the project environment using the following Conda command:
```Python
conda env create -f environment.yml
```

### :pushpin: Data Cleaning and Preparation
___

Minor transformations — renaming one feature and converting all feature names to uppercase — were conducted at this stage.

### :pushpin: Exploratory Data Analysis
___

EDA was implemented in Jupyter notebook. Main findings / conclusions:

- The data is of good quality with no missing or zero values.
  
- The distribution of data for the seven bean varieties is uneven across the dataset. Therefore, stratified sampling used to divide the dataset into test and train sets.</br>
- Since the labels in the dataset are located in a row, shuffling of the data needs to be applied.
- Analysis of the descriptive statistics revealed that the data have a large spread in values and need to be standardized.
- Histograms showed that several features are multimodal. The dataset includes right-skewed, left-skewed, and nearly normally distributed features.
- Multicollinearity was detected in the data. A set of features that are highly correlated with others in the dataset beyond a specified threshold will be identified; this threshold is set at 0.9. I trained and evaluated models on all original 16 features and on data with features removed that exhibit high correlation beyond this threshold.

### :pushpin: Evaluation Metrics
___

The <ins>weighted average F1-score</ins> from the sklearn `classification_report()` was used to evaluate trained models. It provides a single metric that balances both precision and recall, and it adjusts for class imbalance by weighting the F1-score of each class by its presence in the dataset, ensuring a fair representation of model performance across all classes.

### :pushpin: Model training (default parameters) and Results
___

Five supervised machine learning algorithms <ins>**SVM**</ins>, <ins>**DT**</ins>, <ins>**KNN**</ins>, <ins>**RF**</ins> and <ins>**MLP**</ins> were employed to classify beans on this stage. 

To ensure equal representation of each class in the training and testing sets, data shuffling and stratified sampling were employed. All features were standardized using `StandardScaler()`. The initial split is 20% for testing and 80% for training.  

SVM and KNN classifiers with default settings, DT and RF with parameter `random_state` changed to 42 and MLP with `max_iter` parameter changed to 500 and `random_state` changed to 42 were trained on the training set and validated on the test set. Setting the `random_state` parameter to 42 for DT, RF, and MLP classifiers ensures that the randomness built into their processes (like data splitting and initial weight settings) is consistent across runs, making the results reproducible and comparable.

Run following commands to get accuracy reports for each of five classifiers:

1. Train on all original 16 features:
```Python
python3 train_features_all.py
   ```

2. Train on 7 features (9 highly correlated features are pruned):
```Python
python3 train_features_pruned.py
```

Each classifier's accuracy report is saved in the *.txt* file format in the results folder.

### :pushpin: First stage Results
___

F1 scores for each classifier and each class (all features / pruned features), along with the weighted average F1 scores:

|                            |   SVM    |   SVM       |  DT      |  DT     |   KNN    |   KNN   |   RF     |   RF      |   MLP    |  MLP      |
|:---------------------------|:--------:|:-----------:|:--------:|:-------:|:--------:|:-------:|:--------:|:---------:|:--------:|:---------:|
|                            |   All    | Pruned      |   All    | Pruned  |   All    | Pruned  |   All    | Pruned    |   All    | Pruned    |
| **Barbunja**               | 0.9359   | 0.9377      | 0.8942   | 0.9043  | 0.9421   | 0.9302  | 0.9290   | 0.9318    | 0.9256   | 0.9448    |
| **Bombay**                 | 1.0000   | 1.0000      | 1.0000   | 1.0000  | 1.0000   | 1.0000  | 1.0000   | 1.0000    | 1.0000   | 1.0000    |
| **Cali**                   | 0.9559   | 0.9357      | 0.9119   | 0.9186  | 0.9605   | 0.9437  | 0.9388   | 0.9528    | 0.9601   | 0.9633    |
| **Dermason**               | 0.9140   | 0.9151      | 0.8759   | 0.8883  | 0.9104   | 0.8941  | 0.9151   | 0.9222    | 0.9189   | 0.9197    |
| **Horoz**                  | 0.9555   | 0.9570      | 0.9300   | 0.9186  | 0.9450   | 0.9535  | 0.9406   | 0.9515    | 0.9485   | 0.9540    |
| **Seker**                  | 0.9602   | 0.9616      | 0.9365   | 0.9197  | 0.9553   | 0.9514  | 0.9494   | 0.9578    | 0.9602   | 0.9591    |
| **Sira**                   | 0.8759   | 0.8714      | 0.8231   | 0.8138  | 0.8688   | 0.8470  | 0.8692   | 0.8806    | 0.8783   | 0.8828    |
| **Weigh. avg. F1-score**   | 0.9298   | 0.9298 ★    | 0.8932   | 0.8923  | 0.9265   | 0.9154  | 0.9224   | 0.9312 ★  | 0.9327   | 0.9340 ★  |


