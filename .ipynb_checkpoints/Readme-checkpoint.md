
This parent folder should contain the .csv files with the data and two folders: **EDA** and **Model_Training**, the contents of which are outlined below.

__EDA__

This folder contains:
- A notebook containing EDA for on the application_samples.csv data, called `EDA_application_samples.ipynb`
- A notebook containing EDA for on the credit_features.csv data, called  `EDA_credit_features.ipynb`

In those notebooks I explore the data and decide whether and how to use it for a first version of the classifier.
To do so I look at each feature on its, the correlation between the features and the correlation between features and the target, "Success".

The model is then trained in the folder

__Model_Training__

In EDA, I decided on several data preprocessing steps that would be useful. To facilitate eventual production, I have copied these over into a separate notebook called `preprocessing.ipynb`.
In my experience this makes life easier for MLOps or myself, when I deploy a model.

The output from `preprocessing.ipynb` is a dataset ready for training, called `data_4_model.parquet`.
This data is loaded in by `train_and_evaluate_model.ipynb`, where model training and evaluation was done.

I chose an XGBoost classifier because:

- Gradient boosting models like XGBoost or LightGBM are powerful for structured/tabular data, often outperforming Random Forest in terms of predictive accuracy.
- Our dataset has an imbalanced class distribution (e.g., more rejections than successes). Boosting algorithms can handle such imbalances effectively.
- Gradient boosting models capture non-linear relationships, interactions between variables, and can be finely tuned for performance.

This notebook spits out a model file (.json), as well as a data_cleaner and preprocessor objects (.pkl). These are particularly useful for model deployment, as they make it easy to process the data that is to be scored in the same way that the training data.

The code behind these objects sits in the home-made `cleaning.py` module.



### Comments and admissions:

- I was lazy and did not do hyperparameter tuning.
Instead, I used a set of parameters that have worked well for me in the past.
To reduce overfitting on this small dataset, I have brought in L1 and L2 regularization and tried to prevent trees from being too deep.
The choice of XGBoost parameters reflects this.

- I have not gone back and revisited Feature Engineering post model training v1.
I have not dealt with outliers or checked which features can be omitted.
This would be my next step, in conjunction with the SHAPs values.

- I am new to financial data
In the absence of a more elaborate data dictionary I found it hard to decide on how to create new more powerful features.





I haven't addressed
- How you would use the model to make decisions?
- Model governance considerations 
- How the model could be productionized?

Hopefully we can discuss these in an interview :)