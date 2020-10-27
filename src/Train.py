import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
import argparse
import config
import model_dispatcher
import feature_extraction
import joblib


def create_folds(train_path,n_splits):
    # import data
    df = pd.read_csv(train_path,sep = ",")
    # create column "kfold"
    df["kfold"] = -1
    # Randomize data
    df = df.sample(frac = 1).reset_index(drop = True)
    # Values predicted
    y = df.Prediction.values
    # Cross-validaiton using Stratif
    kf = model_selection.StratifiedKFold(n_splits = n_splits)
    # Assign each group of samples to the right fold
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_, "kfold"] = f
    # save the training file in the right fold
    df.to_csv("../input/train_folds.csv",index = False)



def run(train_fold_path,fold,model,extract_method):
    # Import folded training file
    df = pd.read_csv(train_fold_path,sep = ",")
    # convert target variable from categorical to numerical
    df.loc[:,"Prediction"] = df.Prediction.apply(lambda x : 1 if x == "print" else 0)
    # Split all data into Train/valid part
    train_df = df[df.kfold != fold].reset_index(drop = True)
    valid_df = df[df.kfold == fold].reset_index(drop = True)
    # Predictors features
    X_train = train_df.drop("Prediction", axis = 1)
    X_valid = valid_df.drop("Prediction", axis = 1)
    # Variable to predict
    Y_train = train_df.Prediction.values
    Y_valid = valid_df.Prediction.values
    # Features extraction method
    ext_method = feature_extraction.features_dict[extract_method]
    # fitting
    ext_method.fit(X_train.Received_msg)
    # Transform Received msg column
    X_train = ext_method.transform(X_train.Received_msg)
    X_valid = ext_method.transform(X_valid.Received_msg)
    # Classifier
    clf = model_dispatcher.models[model]
    # fitting the classifier
    clf.fit(X_train,Y_train)
    # prediction on the training data
    preds_train = clf.predict(X_train)
    print(f"Fold = {fold} , F1-Score _train = {metrics.f1_score(Y_train,preds_train)}")
    # prediction on the valid data
    preds = clf.predict(X_valid)
    print(preds == Y_valid)
    # F1 score metrics
    f1score = metrics.f1_score(Y_valid,preds)
    print(f"Fold = {fold} , F1-Score = {f1score}")
    # save extract features algorithm
    joblib.dump(ext_method,"../models/count_vec.joblib")
    # save SVM model
    joblib.dump(clf,"../models/model_LogReg.joblib")

    


if __name__ == "__main__":

    create_folds(config.TRAINING_FILE,5)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type = int
    )
    parser.add_argument(
        "--train_path",
        type = str
    )

    parser.add_argument(
        "--model",
        type = str
    )
    parser.add_argument(
        "--extract_method",
        type = str
    )
    args = parser.parse_args()

    run(train_fold_path = args.train_path, fold = args.fold, 
        model = args.model,extract_method = args.extract_method
       )
    

    

