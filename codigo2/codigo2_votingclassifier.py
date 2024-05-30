
import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from joblib import dump

# Constantes:
PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'
PATH3 = 'codigo2/'

# Colunas:
colunas_LGBM = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta',
                b'jetAK8_prunedMass', b'jetAK8_tau21', b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks',
                b'W_rapidity', b'xi1', b'xi2', b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww']

# Função para carregar dados
def load_data(index):
    x_train = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}x_train/x_anomalo_{index}_train.h5", 'r')['treino']),
        columns=colunas_LGBM)
    y_train = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}y_train/y_anomalo_{index}_train.h5", 'r')['treino']))
    x_test = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}x_test/x_anomalo_{index}_test.h5", 'r')['treino']),
        columns=colunas_LGBM)
    y_test = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}y_test/y_anomalo_{index}_test.h5", 'r')['treino']))
    weight = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}weight_anomalo/weight_anomalo_{index}.h5", 'r')['treino']))
    return x_train, y_train, x_test, y_test, weight

# Função para treinar o Voting Classifier
def train_voting_classifier(models, params, X_train, y_train, save_path=None):
    models_params = [(name, model.set_params(**params[name]) if name in params else model) for name, model in models.items()]
    voting_clf = VotingClassifier(estimators=models_params, voting='soft')
    voting_clf.fit(X_train, y_train)
    if save_path:
        dump(voting_clf, save_path)
    accuracy = voting_clf.score(X_train, y_train)
    y_train_proba = voting_clf.predict_proba(X_train)
    loss = log_loss(y_train, y_train_proba)
    return voting_clf, accuracy, loss

# Função para carregar parâmetros de arquivos CSV
def load_params(param_paths, models_to_include):
    params = {}
    for path in param_paths:
        model_name = path.split('/')[-1].split('.')[0]
        if model_name in models_to_include:
            df = pd.read_csv(path)
            model_params = df.to_dict(orient='records')[0]
            params[model_name] = model_params
    return params

# Definir os modelos
logistregrecion = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1)
)
xgbc = XGBClassifier()
rfc = RandomForestClassifier()
lgbmc = LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31
)

models = {'lr': logistregrecion, 'xgbc': xgbc, 'rfc': rfc, 'lgbm': lgbmc}

# Loop de 1 a 8
for i in range(1, 9):
    print(f"Treinando para anomalo_{i}")
    param_paths = [
        f"{PATH}{PATH3}xgbc/melhores_parametros_anomalo_{i}.csv",
        f"{PATH}{PATH3}rfc/melhores_parametros_rf_anomalo_{i}.csv"
    ]
    models_to_include = ['xgbc', 'rfc']
    params = load_params(param_paths, models_to_include)
    x_train, y_train, _, _, _ = load_data(i)
    y_train = y_train.values.ravel()
    save_path = f"{PATH}{PATH3}voting_classifier_anomalo_{i}.joblib"
    voting_clf, accuracy, loss = train_voting_classifier(models, params, x_train, y_train, save_path)
    print(f"anômalo_{i} - Accuracy:", accuracy)
    print(f"anômalo_{i} - Log Loss:", loss)

