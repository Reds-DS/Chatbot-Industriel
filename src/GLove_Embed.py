import io
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel,delayed
from tqdm import tqdm
from sklearn import svm
from sklearn import model_selection
import xgboost
from sklearn import tree,ensemble
from sklearn.neighbors import KNeighborsClassifier

# Accuracy train : 83%
# Accuracy test : 72%
# Overfitting
# Problem with phrase that contains doc pages but not impress/prints


def load_vectors(fname):
    fin = io.open(
        fname,
        "r",
        encoding = "utf-8",
        newline = "\n",
        errors = "ignore"
    )

    #n, d = map(int, fin.readline().split())
    data = {}
    for line in fin : 
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]

    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])
    
    if len(M) == 0:
        return np.zeros(50)

    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v**2).sum())

def create_folds(train_path,n_splits):
    """
    Function Creating new file, with column fold which help us to split train/Valid data
    train_path : path to the training file dataset
    n_splits : number of splits for cross-validation
    """
    df = pd.read_csv(train_path, sep = ",") 

    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits = n_splits)

    for fold,(trn_,val_) in enumerate(kf.split(X = df,y = df.Prediction.values)):
        df.loc[val_,"kfold"] = fold 
    
    df.to_csv("../input/train_folded.csv", index = False)



def run(fold):
    # Import Training data
    df = pd.read_csv("../input/train_folded.csv", sep = ",")

    # Convert column prediction from categorical to numerical
    df.Prediction = df.Prediction.apply(lambda x : 1 if x == "print" else 0)

    # Split the data into train/valid dataset
    train_df = df[df.kfold != fold].reset_index(drop = True)
    test_df = df[df.kfold == fold].reset_index(drop = True)


    print("----- Load embeddings --------")

    # Pre-trained Glove vectors from Stanford University Website
    embeddings = load_vectors("../input/glove.6B.50d.txt")
    print("Loading Complete!")

    print("Creating sentence vectors ... ")
    vectors = []
    vectors_test = []

    # Creating sentences vectors 
    for msg in train_df.Received_msg.values:
        vectors.append(
            sentence_to_vec(
                s = msg, embedding_dict = embeddings,
                stop_words = [],tokenizer = word_tokenize
            )
        )
    for msg_test in test_df.Received_msg.values:
        vectors_test.append(
            sentence_to_vec(
                s = msg_test, embedding_dict = embeddings,
                stop_words = [],tokenizer = word_tokenize
            )
        )  
    print("Finish !")
    
    # Convert vectors to array
    vectors = np.array(vectors)
    vectors_test = np.array(vectors_test)


    # Stack verctors
    X_train = np.stack(vectors,axis = 0)
    Y_train = train_df.Prediction.values

    
    print("Initializing the model .. ")
    model = svm.SVC(C = 0.5, kernel = "linear")
    #model = linear_model.LogisticRegression()
    #model = xgboost.XGBClassifier()
    #model = tree.DecisionTreeClassifier()
    #model = ensemble.RandomForestClassifier()
    #model = KNeighborsClassifier()
    print("Fitting the model .. ")
    model.fit(X_train, Y_train)
    
    print("Training SVM ..")
    preds_train  = model.predict(X_train)
    accuracy = metrics.f1_score(train_df.Prediction, preds_train)
    print(f"F1-score for Train = {accuracy}")

    print("Prediction over a new dataset : ")
    preds = model.predict(np.stack(vectors_test, axis = 0))
    #print(preds)
    print("Prediction finished !")
    
    accuracy = metrics.f1_score(test_df.Prediction, preds)
    print(f"F1-score for test = {accuracy}")
    print("\n")
    print(preds != test_df.Prediction )

if __name__ == "__main__":

    create_folds("../input/train.csv", 5)
    run(3)

    