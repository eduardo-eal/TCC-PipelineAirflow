import mlflow
import mlflow.xgboost
import argparse
import pandas as pd
from urllib.parse import urlparse
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("max_depth", help="parametro max_depth", default=6,type=int)
parser.add_argument("n_estimators", help="parametro n_estimators", default=100, type=int)
parser.add_argument("colsample_bytree", help="parametro colsample_bytree", default=0.8, type=float)
parser.add_argument("subsample", help="parametro subsample", default=0.8, type=float)
parser.add_argument("nthread", help="parametro nthread", default=10, type=int)
parser.add_argument("learning_rate", help="parametro learning_rate", default=0.3, type=float)

args = parser.parse_args()

pathScriptFeatureStore = "./featurestore"

infile_X_treino_tfidf = open(pathScriptFeatureStore+'/X_treino_tfidf.sm','rb')
X_treino_tfidf = pickle.load(infile_X_treino_tfidf)
infile_X_treino_tfidf.close()

infile_X_teste_tfidf = open(pathScriptFeatureStore+'/X_teste_tfidf.sm','rb')
X_teste_tfidf = pickle.load(infile_X_teste_tfidf)
infile_X_teste_tfidf.close()

y_treino = pd.read_csv(pathScriptFeatureStore+'/y_treino.csv')

y_teste = pd.read_csv(pathScriptFeatureStore+'/y_teste.csv')

def get_metrics(y_teste, y_predicao):  
    precision = round(precision_score(y_teste, y_predicao, pos_label=1, average='macro'),4)             
    recall = round(recall_score(y_teste, y_predicao, pos_label=1, average='macro'),4)
    f1 = round(f1_score(y_teste, y_predicao, pos_label=1, average='macro'),4)
    accuracy = round(accuracy_score(y_teste, y_predicao),4)
    return accuracy, precision, recall, f1

try:
    idExperiment = mlflow.create_experiment('tcc_xgboost')
except:
    idExperiment = mlflow.get_experiment_by_name('tcc_xgboost').experiment_id

with mlflow.start_run(experiment_id=idExperiment):
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("colsample_bytree", args.colsample_bytree)
    mlflow.log_param("subsample", args.subsample)
    mlflow.log_param("nthread", args.nthread)
    mlflow.log_param("learning_rate", args.learning_rate)


    modelo_xgboost = xgb.XGBClassifier(max_depth=args.max_depth, n_estimators=args.n_estimators, colsample_bytree=args.colsample_bytree, 
	                subsample=args.subsample, nthread=args.nthread, learning_rate=args.learning_rate)
    modelo_xgboost.fit(X_treino_tfidf, y_treino)


    y_predicao = modelo_xgboost.predict(X_teste_tfidf)

    accuracy, precision, recall, f1 = get_metrics(y_teste, y_predicao)

    mlflow.log_metric("accuracy_score", accuracy)


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.xgboost.log_model(modelo_xgboost, "model", registered_model_name='Modelo_XGBoost')
    else:
        mlflow.xgboost.log_model(modelo_xgboost, "model")


