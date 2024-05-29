# ----------------------------------------------------------------------------------------------------------
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump
import h5py
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------------

# Constantes:

PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'
PATH3 = 'codigo2/v2/'

#----------------------------------------------------------------------------------------------------------
# Colunas:

colunas_LGBM = [
    b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta',
    b'jetAK8_prunedMass', b'jetAK8_tau21', b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks',
    b'W_rapidity', b'xi1', b'xi2', b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww'
]

#----------------------------------------------------------------------------------------------------------
# Loop de 1 a 8
for i in range(1, 9):
    # Carregar o arquivo correspondente e criar o DataFrame
    # x_train:
    globals()[f"x_anomalo_{i}_train"] = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}x_train/x_anomalo_{i}_train.h5", 'r')['treino']),
        columns=colunas_LGBM)
    # y_train:
    globals()[f"y_anomalo_{i}_train"] = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}y_train/y_anomalo_{i}_train.h5", 'r')['treino']))
    # x_test:
    globals()[f"x_anomalo_{i}_test"] = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}x_test/x_anomalo_{i}_test.h5", 'r')['treino']),
        columns=colunas_LGBM)
    # y_test:
    globals()[f"y_anomalo_{i}_test"] = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}y_test/y_anomalo_{i}_test.h5", 'r')['treino']))
    # pesos:
    globals()[f"weight_anomalo_{i}"] = pd.DataFrame(
        np.array(h5py.File(f"{PATH}{PATH2}weight_anomalo/weight_anomalo_{i}.h5", 'r')['treino']))

# ----------------------------------------------------------------------------------------------------------
# Funções de LOSS

def loss_logistic_regression(y_true, y_pred_proba):
    from sklearn.metrics import log_loss
    return log_loss(y_true, y_pred_proba)


def loss_random_forest_classifier(y_true, y_pred_proba):
    from sklearn.metrics import log_loss
    return log_loss(y_true, y_pred_proba)


def loss_xgboost_classifier(y_true, y_pred_proba):
    from sklearn.metrics import log_loss
    return log_loss(y_true, y_pred_proba)


# ----------------------------------------------------------------------------------------------------------
# Função para treinar e salvar modelo de Logistic Regression
def treinar_modelo_e_salvar_lr(x_train, y_train, x_test, y_test, caminho_salvar, caminho_loss):
    if len(y_train.shape) > 1:
        y_train = y_train.values.ravel()

    lor_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1)
    )

    lor_clf.fit(x_train, y_train)
    print("Número de iterações utilizadas:", lor_clf.named_steps['logisticregression'].n_iter_)

    y_pred_proba = lor_clf.predict_proba(x_test)[:, 1]
    loss = loss_logistic_regression(y_test, y_pred_proba)
    print("Loss no conjunto de teste:", loss)

    os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)  # Criar diretório se não existir
    dump(lor_clf, caminho_salvar)

    # Salvar loss em CSV
    with open(caminho_loss, 'a') as f:
        f.write(f'Logistic Regression, {loss}\n')


# ----------------------------------------------------------------------------------------------------------
# Função para treinar e salvar modelo de Random Forest Classifier
def treinar_modelo_e_salvar_rfc(x_train, y_train, x_test, y_test, caminho_modelo, caminho_parametros, caminho_loss):
    if len(y_train.shape) > 1:
        y_train = y_train.values.ravel()

    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy',
                               n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_rf_clf = grid_search.best_estimator_
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)  # Criar diretório se não existir
    dump(best_rf_clf, caminho_modelo)

    melhores_parametros = pd.DataFrame(grid_search.best_params_, index=[0])
    melhores_parametros.to_csv(caminho_parametros, index=False)

    y_pred_proba_rf = best_rf_clf.predict_proba(x_test)[:, 1]
    loss = loss_random_forest_classifier(y_test, y_pred_proba_rf)
    print("Loss no conjunto de teste:", loss)

    with open(caminho_loss, 'a') as f:
        f.write(f'Random Forest, {loss}\n')


# ----------------------------------------------------------------------------------------------------------
# Função para treinar e salvar modelo de XGBoost Classifier
def treinar_modelo_e_salvar_xgbc(x_train, y_train, x_test, y_test, caminho_modelo, caminho_parametros, caminho_csv,
                                 caminho_loss):
    xgb_clf = XGBClassifier()

    param_grid = {
        'n_estimators': [500, 1000, 1500, 2000, 2500],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [6, 10, 15]
    }

    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_xgb_clf = grid_search.best_estimator_
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)  # Criar diretório se não existir
    dump(best_xgb_clf, caminho_modelo)

    melhores_parametros = pd.DataFrame(grid_search.best_params_, index=[0])
    melhores_parametros.to_csv(caminho_parametros, index=False)

    y_pred_proba_xgb = best_xgb_clf.predict_proba(x_test)[:, 1]
    loss = loss_xgboost_classifier(y_test, y_pred_proba_xgb)
    print("Loss no conjunto de teste:", loss)

    resultados = pd.DataFrame({"Probabilidade Classe 1": y_pred_proba_xgb})
    resultados.to_csv(caminho_csv, mode='a', index=False, header=not os.path.exists(caminho_csv))

    with open(caminho_loss, 'a') as f:
        f.write(f'XGBoost, {loss}\n')


# ----------------------------------------------------------------------------------------------------------
# Loop de 1 a 8 para rodar a função com as variações
for i in range(1, 9):
    x_train = globals()[f'x_anomalo_{i}_train']
    y_train = globals()[f'y_anomalo_{i}_train']
    x_test = globals()[f'x_anomalo_{i}_test']
    y_test = globals()[f'y_anomalo_{i}_test']

    model_filepath_lr = os.path.join(PATH, PATH3, f'lr/lor_clf_binario_randomico_anomalo_{i}_DataDriven.joblib')
    model_filepath_rfc = os.path.join(PATH, PATH3, f'rfc/best_rf_clf_binario_randomico_anomalo_{i}_DataDriven.joblib')
    best_params_filepath_rfc = os.path.join(PATH, PATH3, f'melhores_parametros_rf_anomalo_{i}.csv')
    model_filepath_xgbc = os.path.join(PATH, PATH3,
                                       f'xgbc/best_xgb_clf_binario_randomico_anomalo_{i}_DataDriven.joblib')
    best_params_filepath_xgbc = os.path.join(PATH, PATH3, f'xgbc/melhores_parametros_anomalo_{i}.csv')
    all_params_filepath_xgbc = os.path.join(PATH, PATH3, f'xgbc/parametros.csv')
    loss_filepath = os.path.join(PATH, PATH3, f'loss_anomalo_{i}.csv')

    treinar_modelo_e_salvar_lr(x_train, y_train, x_test, y_test, model_filepath_lr, loss_filepath)
    treinar_modelo_e_salvar_rfc(x_train, y_train, x_test, y_test, model_filepath_rfc, best_params_filepath_rfc,
                                loss_filepath)
    treinar_modelo_e_salvar_xgbc(x_train, y_train, x_test, y_test, model_filepath_xgbc, best_params_filepath_xgbc,
                                 all_params_filepath_xgbc, loss_filepath)
