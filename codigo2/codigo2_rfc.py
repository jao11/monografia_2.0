# Instalando bibliotecas e fazendo imports necessarios:

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


import h5py
import numpy as np
import pandas as pd

# Constantes:

PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'
PATH3 = 'codigo2/'
#----------------------------------------------------------------------------------------------------------
# Colunas:

colunas_LGBM = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta', b'jetAK8_prunedMass', b'jetAK8_tau21',
            b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'W_rapidity', b'xi1', b'xi2',  b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww']
#----------------------------------------------------------------------------------------------------------

# Abrindo os arquivos preparados no código 1:

# x_train:

x_anomalo_1_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_1_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_2_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_2_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_3_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_3_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_4_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_4_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_5_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_5_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_6_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_6_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_7_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_7_train.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_8_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_train/x_anomalo_8_train.h5','r')['treino']),columns=colunas_LGBM)

# y_train:

y_anomalo_1_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_1_train.h5','r')['treino']))#,columns=colunas_LGBM)
y_anomalo_2_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_2_train.h5','r')['treino']))
y_anomalo_3_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_3_train.h5','r')['treino']))
y_anomalo_4_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_4_train.h5','r')['treino']))
y_anomalo_5_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_5_train.h5','r')['treino']))
y_anomalo_6_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_6_train.h5','r')['treino']))
y_anomalo_7_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_7_train.h5','r')['treino']))
y_anomalo_8_train = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_train/y_anomalo_8_train.h5','r')['treino']))

# x_test:

x_anomalo_1_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_1_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_2_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_2_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_3_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_3_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_4_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_4_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_5_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_5_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_6_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_6_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_7_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_7_test.h5','r')['treino']),columns=colunas_LGBM)
x_anomalo_8_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'x_test/x_anomalo_8_test.h5','r')['treino']),columns=colunas_LGBM)

# y_test:

y_anomalo_1_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_1_test.h5','r')['treino']))#,columns=colunas_LGBM)
y_anomalo_2_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_2_test.h5','r')['treino']))
y_anomalo_3_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_3_test.h5','r')['treino']))
y_anomalo_4_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_4_test.h5','r')['treino']))
y_anomalo_5_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_5_test.h5','r')['treino']))
y_anomalo_6_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_6_test.h5','r')['treino']))
y_anomalo_7_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_7_test.h5','r')['treino']))
y_anomalo_8_test = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'y_test/y_anomalo_8_test.h5','r')['treino']))


# pesos:

weight_anomalo_1 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_1.h5','r')['treino']))#,columns=colunas_LGBM)
weight_anomalo_2 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_2.h5','r')['treino']))
weight_anomalo_3 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_3.h5','r')['treino']))
weight_anomalo_4 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_4.h5','r')['treino']))
weight_anomalo_5 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_5.h5','r')['treino']))
weight_anomalo_6 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_6.h5','r')['treino']))
weight_anomalo_7 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_7.h5','r')['treino']))
weight_anomalo_8 = pd.DataFrame(np.array(h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_8.h5','r')['treino']))


# Random Forest:

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def treinar_modelo_e_salvar_rf(x_train, y_train, x_test, caminho_modelo, caminho_parametros):
    # Certifique-se de que y_train seja um array 1D
    if len(y_train.shape) > 1:
        y_train = y_train.values.ravel()

    # Definir os parâmetros que serão testados
    param_grid = {
        'n_estimators': [100, 500, 1000],     # Testar diferentes números de árvores
        'max_depth': [None, 10, 20],        # Testar diferentes profundidades máximas das árvores
        'min_samples_split': [2, 5, 10],       # Testar diferentes valores mínimos de amostras para divisão
        'min_samples_leaf': [1, 2, 4]          # Testar diferentes valores mínimos de amostras em folhas
    }

    # Criar um objeto GridSearchCV
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Treinar o modelo com os dados de treinamento
    grid_search.fit(x_train, y_train)

    # Obter o melhor modelo encontrado
    best_rf_clf = grid_search.best_estimator_

    # Salvar o melhor modelo treinado
    dump(best_rf_clf, caminho_modelo)

    # Criar um DataFrame com os melhores parâmetros
    melhores_parametros = pd.DataFrame(grid_search.best_params_, index=[0])

    # Salvar os melhores parâmetros em um arquivo CSV
    melhores_parametros.to_csv(caminho_parametros, index=False)

    # Fazer previsões de probabilidade no conjunto de teste usando o melhor modelo encontrado
    y_pred_proba_rf = best_rf_clf.predict_proba(x_test)

    # Extrair as probabilidades da classe 1 (índice 1)
    probabilidade_classe_1 = y_pred_proba_rf[:, 1]

    # Exibir as primeiras 10 probabilidades em formato de porcentagem
    print("Probabilidades de pertencer à classe 1:", ["{:.2f}%".format(probabilidade * 100) for probabilidade in probabilidade_classe_1[:10]])


# Uso
treinar_modelo_e_salvar_rf(x_anomalo_1_train, y_anomalo_1_train, x_anomalo_1_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_1_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_1.csv')
treinar_modelo_e_salvar_rf(x_anomalo_2_train, y_anomalo_2_train, x_anomalo_2_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_2_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_2.csv')
treinar_modelo_e_salvar_rf(x_anomalo_3_train, y_anomalo_3_train, x_anomalo_3_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_3_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_3.csv')
treinar_modelo_e_salvar_rf(x_anomalo_4_train, y_anomalo_4_train, x_anomalo_4_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_4_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_4.csv')
treinar_modelo_e_salvar_rf(x_anomalo_5_train, y_anomalo_5_train, x_anomalo_5_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_5_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_5.csv')
treinar_modelo_e_salvar_rf(x_anomalo_6_train, y_anomalo_6_train, x_anomalo_6_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_6_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_6.csv')
treinar_modelo_e_salvar_rf(x_anomalo_7_train, y_anomalo_7_train, x_anomalo_7_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_7_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_7.csv')
treinar_modelo_e_salvar_rf(x_anomalo_8_train, y_anomalo_8_train, x_anomalo_8_test, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_8_DataDriven.joblib', PATH + PATH3 + 'melhores_parametros_rf_anomalo_8.csv')



'''
# Definir os parâmetros que serão testados
param_grid = {
    'n_estimators': [100, 500, 1000],     # Testar diferentes números de árvores
    'max_depth': [None, 10, 20],        # Testar diferentes profundidades máximas das árvores
    'min_samples_split': [2, 5, 10],       # Testar diferentes valores mínimos de amostras para divisão
    'min_samples_leaf': [1, 2, 4]          # Testar diferentes valores mínimos de amostras em folhas
}

# Criar um objeto GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Treinar o modelo com os dados de treinamento
grid_search.fit(x_anomalo_3_train, y_anomalo_3_train)

# Obter o melhor modelo encontrado
best_rf_clf = grid_search.best_estimator_

# Imprimir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Fazer previsões de probabilidade no conjunto de teste usando o melhor modelo encontrado
y_pred_proba_rf_3 = best_rf_clf.predict_proba(x_anomalo_3_test)

# Extrair as probabilidades da classe 1 (índice 1)
probabilidade_classe_1 = y_pred_proba_rf_3[:, 1]

# Exibir as primeiras 10 probabilidades em formato de porcentagem
print("Probabilidades de pertencer à classe 1:")
for probabilidade in probabilidade_classe_1[:10]:
    print("{:.2f}%".format(probabilidade * 100))

# Salvar o melhor modelo treinado
dump(best_rf_clf, PATH + PATH3 + 'rfc/best_rf_clf_binario_randomico_anomalo_3_DataDriven.joblib')
'''


