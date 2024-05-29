# Instalando bibliotecas e fazendo imports necessarios:

import h5py
import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump

# Constantes:

PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'
PATH3 = 'codigo2/'
# ----------------------------------------------------------------------------------------------------------
# Colunas:

colunas_LGBM = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta',
                b'jetAK8_prunedMass', b'jetAK8_tau21',
                b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'W_rapidity', b'xi1', b'xi2', b'Mpps', b'Ypps',
                b'Mww/Mpps', b'Ypps-Yww']
# ----------------------------------------------------------------------------------------------------------

# Abrindo os arquivos preparados no código 1:

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

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# LGBM Classifier:

# Versao usada pelo Macedo.
'''
iris = load_iris()
lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary') #, random_state=0)

param_dist = {
    'num_leaves': [10, 20, 30, 40, 50],  # Número máximo de folhas em uma árvore
    'learning_rate': uniform(loc=0, scale=1),  # Taxa de aprendizado
    'n_estimators': [100, 200, 300, 400, 500, 1000],  # Número de estimadores (árvores) no modelo
    'subsample': uniform(loc=0.5, scale=0.5),  # Subsample ratio of the training instances
    'colsample_bytree': uniform(loc=0.5, scale=0.5),  # Subsample ratio of columns when constructing each tree
}

clf = RandomizedSearchCV(lgbm, param_distributions=param_dist, random_state=0)
search = clf.fit(iris.data, iris.target)
search.best_params_
'''


def treinar_modelo_e_salvar_lgbm(x_train, y_train, x_test, caminho_modelo, caminho_parametros):
    param_search = {'num_leaves': [32, 64],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 500, 1000],
                    'min_child_samples': [20, 50, 100],  # Reduzimos o valor mínimo de amostras em cada folha
                    'subsample': [0.4, 0.8],  # Testamos valores mais baixos de subsample
                    'colsample_bytree': [0.3, 1.0],  # Testamos valores mais baixos de colsample_bytree
                    'is_unbalance': [False],  # depois troca para o False
                    'boosting_type': ['gbdt'],
                    }

    '''
    param_search = {'num_leaves': [64, 128, 256],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 1000, 2000, 5000],
                    'min_child_samples': [20, 80, 120],
                    'subsample': [0.4, 0.9],
                    'colsample_bytree': [0.3, 1],
                    'is_unbalance': [True],
                    'boosting_type': ['gbdt'],
                    }
    '''
    cv = 3
    scoring = 'f1'

    model0 = LGBMClassifier(boosting_type='gbdt', objective='binary')
    # RandomSearch procura o nome depois
    grid_search = GridSearchCV(estimator=model0, param_grid=param_search, scoring=scoring, cv=cv, verbose=2, n_jobs=-1)

    # Convertendo y_train para uma matriz unidimensional
    y_train = y_train.values.ravel()

    # Treinar o modelo com os dados de treinamento
    grid_search.fit(x_train, y_train)

    # Obter o melhor modelo encontrado
    best_lgbm_clf = grid_search.best_estimator_

    # Salvar o melhor modelo treinado
    dump(best_lgbm_clf, caminho_modelo)

    # Criar um DataFrame com os melhores parâmetros
    melhores_parametros = pd.DataFrame(grid_search.best_params_, index=[0])

    # Salvar os melhores parâmetros em um arquivo CSV
    melhores_parametros.to_csv(caminho_parametros, index=False)

    # Fazer previsões de probabilidade no conjunto de teste usando o melhor modelo encontrado
    y_pred_proba_lgbm = best_lgbm_clf.predict_proba(x_test)

    # Extrair as probabilidades da classe 1 (índice 1)
    probabilidade_classe_1 = y_pred_proba_lgbm[:, 1]

    # Imprimir as probabilidades de pertencer à classe 1
    print("Probabilidades de pertencer à classe 1:", probabilidade_classe_1)


# treinar_modelo_e_salvar_lgbm(x_anomalo_1_train, y_anomalo_1_train, x_anomalo_1_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_1_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_1.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_2_train, y_anomalo_2_train, x_anomalo_2_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_2_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_2.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_3_train, y_anomalo_3_train, x_anomalo_3_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_3_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_3.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_4_train, y_anomalo_4_train, x_anomalo_4_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_4_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_4.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_5_train, y_anomalo_5_train, x_anomalo_5_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_5_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_5.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_6_train, y_anomalo_6_train, x_anomalo_6_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_6_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_6.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_7_train, y_anomalo_7_train, x_anomalo_7_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_7_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_7.csv')
# treinar_modelo_e_salvar_lgbm(x_anomalo_8_train, y_anomalo_8_train, x_anomalo_8_test,
#                             PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_8_DataDriven.joblib',
#                             PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_8.csv')


def treinar_modelo_e_salvar_lgbm_random(x_train, y_train, x_test, caminho_modelo, caminho_parametros, caminho_tempo):
    param_dist = {
        'num_leaves': [10, 20, 30, 40, 50],  # Número máximo de folhas em uma árvore
        'learning_rate': uniform(loc=0, scale=1),  # Taxa de aprendizado
        'n_estimators': [100, 200, 300, 400, 500, 1000],  # Número de estimadores (árvores) no modelo
        'subsample': uniform(loc=0.5, scale=0.5),  # Subsample ratio of the training instances
        'colsample_bytree': uniform(loc=0.5, scale=0.5),  # Subsample ratio of columns when constructing each tree
    }

    cv = 3
    scoring = 'f1'
    n_iter = 20  # Número de combinações de hiperparâmetros testadas

    model0 = LGBMClassifier(boosting_type='gbdt', objective='binary')

    random_search = RandomizedSearchCV(estimator=model0, param_distributions=param_dist,
                                       n_iter=n_iter, scoring=scoring, cv=cv, verbose=2, n_jobs=-1)

    # Convertendo y_train para uma matriz unidimensional
    y_train = y_train.values.ravel()

    # Treinar o modelo com os dados de treinamento
    start_time = time.time()
    random_search.fit(x_train, y_train)
    end_time = time.time()

    # Obter o melhor modelo encontrado
    best_lgbm_clf = random_search.best_estimator_

    # Salvar o melhor modelo treinado
    dump(best_lgbm_clf, caminho_modelo)

    # Criar um DataFrame com os melhores parâmetros
    melhores_parametros = pd.DataFrame(random_search.best_params_, index=[0])

    # Salvar os melhores parâmetros em um arquivo CSV
    os.makedirs(os.path.dirname(caminho_parametros), exist_ok=True)
    melhores_parametros.to_csv(caminho_parametros, index=False)

    # Tempo total de treinamento
    total_time = end_time - start_time

    # Salvar o tempo de treinamento em um arquivo CSV
    os.makedirs(os.path.dirname(caminho_tempo), exist_ok=True)
    with open(caminho_tempo, 'w') as f:
        f.write(f"total_time={total_time:.2f} seconds\n")

    # Fazer previsões de probabilidade no conjunto de teste usando o melhor modelo encontrado
    y_pred_proba_lgbm = best_lgbm_clf.predict_proba(x_test)

    # Extrair as probabilidades da classe 1 (índice 1)
    probabilidade_classe_1 = y_pred_proba_lgbm[:, 1]

    # Imprimir as probabilidades de pertencer à classe 1
    print("Probabilidades de pertencer à classe 1:", probabilidade_classe_1)


# Exemplo de uso
#treinar_modelo_e_salvar_lgbm_random(x_anomalo_1_train, y_anomalo_1_train, x_anomalo_1_test,
#                                    PATH + PATH3 + 'lgbmc/best_lgbm_clf_binario_randomico_anomalo_1_DataDriven.joblib',
#                                    PATH + PATH3 + 'lgbmc/melhores_parametros_lgbm_anomalo_1.csv',
#                                    PATH + PATH3 + 'lgbmc/tempo_treinamento_anomalo_1.csv')
import os
import pandas as pd
import joblib
import time
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import numpy as np

def treinar_modelos_e_salvar_lgbm_random(x_train, y_train, x_test, caminho_parametros_csv, caminho_tempo_csv, diretorio_modelos):
    # Criar o diretório se não existir
    if not os.path.exists(diretorio_modelos):
        os.makedirs(diretorio_modelos)

    # Lista de caminhos para os modelos
    caminhos_modelo = [
        os.path.join(diretorio_modelos, f'best_lgbm_clf_binario_randomico_anomalo_1_modelo_{i+1}.joblib')
        for i in range(5)
    ]

    # Parâmetros para o RandomizedSearchCV
    param_dist = {
        'num_leaves': range(20, 150, 10),
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8]
    }

    resultados = []

    for i in range(5):
        start_time = time.time()

        lgbm = LGBMClassifier()
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_dist,
            n_iter=50,
            scoring='f1',
            cv=5,
            verbose=1,
            n_jobs=-1
        )

        # Converter y_train e y_test para vetores 1D
        y_train_1d = np.ravel(y_train)

        random_search.fit(x_train, y_train_1d)

        best_model = random_search.best_estimator_
        joblib.dump(best_model, caminhos_modelo[i])

        best_params = random_search.best_params_
        best_params['model'] = f'model_{i+1}'
        resultados.append(best_params)

        end_time = time.time()
        total_time = end_time - start_time

        with open(caminho_tempo_csv, 'a') as f:
            f.write(f"Model {i+1}: {total_time / 60:.2f} minutes\n")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(caminho_parametros_csv, index=False)


diretorio_modelos = PATH + PATH3 + 'lgbmc/randomico_anomalo_1_modelo'
treinar_modelos_e_salvar_lgbm_random(
    x_anomalo_1_train, y_anomalo_1_train, x_anomalo_1_test,
    PATH + PATH3 + 'lgbm/melhores_parametros_lgbm_anomalo_1.csv',
    PATH + PATH3 + 'lgbm/tempo_treinamento_anomalo_1.csv',
    diretorio_modelos
)
