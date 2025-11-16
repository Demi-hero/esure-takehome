code-base to run EDA and Model Calibration on a kaggle insurance dataset. 

Kaggle data can be found here :https://www.kaggle.com/datasets/ycanario/home-insurance/data

The enviroment file for this is esure.yaml

To run: 
## EDA 
Initial_eda.ipynb will load in the data set and perform univariate analysis as well as statistical tests on the variables

# Model Training 
models.ipynb will load in the processed data from initial_eda and create an xgboost model based off of the data fed in and save it to the model folder. 

## Inference
If you want to use the model that has been generated. You need to supply the path to both the data set and the model 

Usage example:
    python main.py --input data.csv --model Models/2025-11-15/xgb_model.json

