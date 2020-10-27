from sklearn import linear_model,tree
from sklearn import svm
from sklearn import naive_bayes
import xgboost
from sklearn import ensemble
from sklearn import neighbors

models = {
    "decision_tree_gini" : tree.DecisionTreeClassifier(
        criterion = "gini"
    ),
    "Logistic_Regression" : linear_model.LogisticRegression(
        C = 5, penalty = "l2"
    ),
    "SVC" : svm.SVC(
        C = 5, kernel = "rbf",random_state = 0
    ),
    "Xgboost" : xgboost.XGBClassifier(n_estimators = 200),

    "Random_Forest" : ensemble.RandomForestClassifier(),

    "KNN" : neighbors.KNeighborsClassifier(n_neighbors = 5)

    
}

