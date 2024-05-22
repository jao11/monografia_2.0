# Instalando bibliotecas e fazendo imports necessarios:

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression:



def treinar_modelo_e_salvar(x_train, y_train, x_test, caminho_salvar):
    # Certifique-se de que y_train seja uma matriz 1D (se necessário)
    if len(y_train.shape) > 1:
        y_train = y_train.values.ravel()

    # Definir o modelo LogisticRegression para classificação
    lor_clf = make_pipeline(
        StandardScaler(),  # Normalização dos dados
        LogisticRegression(
            solver='lbfgs',      # Algoritmo de otimização
            max_iter=1000,       # Número máximo de iterações
            C=0.1                # Parâmetro de regularização
        )
    )

    # Treinar o modelo com os dados de treinamento
    lor_clf.fit(x_train, y_train)

    # Verificar o número de iterações utilizadas
    print("Número de iterações utilizadas:", lor_clf.named_steps['logisticregression'].n_iter_)

    # Fazer previsões no conjunto de teste
    y_predict = lor_clf.predict(x_test)

    # Calcular a probabilidade de pertencer a classe 1
    y_pred_proba = lor_clf.predict_proba(x_test)[:, 1]

    # Imprimir as probabilidades de pertencer à classe 1 em formato de porcentagem
    #print("Probabilidades de pertencer à classe 1:", ["{:.2f}%".format(probabilidade * 100) for probabilidade in y_pred_proba])

    # Salvar o modelo treinado
    dump(lor_clf, caminho_salvar)

# Exemplo de uso da função
treinar_modelo_e_salvar(x_anomalo_1_train, y_anomalo_1_train, x_anomalo_1_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_1_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_2_train, y_anomalo_2_train, x_anomalo_2_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_2_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_3_train, y_anomalo_3_train, x_anomalo_3_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_3_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_4_train, y_anomalo_4_train, x_anomalo_4_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_4_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_5_train, y_anomalo_5_train, x_anomalo_5_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_5_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_6_train, y_anomalo_6_train, x_anomalo_6_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_6_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_7_train, y_anomalo_7_train, x_anomalo_7_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_7_DataDriven.joblib')
treinar_modelo_e_salvar(x_anomalo_8_train, y_anomalo_8_train, x_anomalo_8_test, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_8_DataDriven.joblib')


'''
# Certifique-se de que y_anomalo_3_train seja uma matriz 1D (se necessário)
if len(y_anomalo_3_train.shape) > 1:
    y_anomalo_3_train = y_anomalo_3_train.values.ravel()

# Definir o modelo LogisticRegression para classificação
lor_clf_3 = make_pipeline(
    StandardScaler(),  # Normalização dos dados
    LogisticRegression(
        solver='lbfgs',      # Algoritmo de otimização
        max_iter=1000,       # Número máximo de iterações
        C=0.1                # Parâmetro de regularização
    )
)

# Treinar o modelo com os dados de treinamento
lor_clf_3.fit(x_anomalo_3_train, y_anomalo_3_train)

# Verificar o número de iterações utilizadas
print("Número de iterações utilizadas:", lor_clf_3.named_steps['logisticregression'].n_iter_)

# Fazer previsões no conjunto de teste
y_predict_lor_3 = lor_clf_3.predict(x_anomalo_3_test)

# Calcular a probabilidade de pertencer a classe 1
y_pred_proba_lor_3 = lor_clf_3.predict_proba(x_anomalo_3_test)[:, 1]

# Imprimir as probabilidades de pertencer à classe 1 em formato de porcentagem
print("Probabilidades de pertencer à classe 1:", ["{:.2f}%".format(probabilidade * 100) for probabilidade in y_pred_proba_lor_3])

# Salvar o modelo treinado
dump(lor_clf_3, PATH + PATH3 + 'lr/lor_clf_binario_randomico_anomalo_3_DataDriven.joblib')
'''

