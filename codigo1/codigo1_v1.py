# Instalando bibliotecas e fazendo imports necessarios:

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os
import h5py

import numpy as np
import pandas as pd

# Constantes facilitadoras e colunas:

raiz_s = 13000  # GeV - consante que representa a energia na colisão dos prótons
teste_size = 0.3  # ??
PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'

coluna = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta',
          b'jetAK8_prunedMass', b'jetAK8_tau21',
          b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'PUWeight', b'W_rapidity', b'btag', b'xi1', b'xi2',
          b'ismultirp1', b'ismultirp2',
          b'Norm', b'weight', b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww']

coluna_LGBM = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta',
               b'jetAK8_prunedMass', b'jetAK8_tau21',
               b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'W_rapidity', b'xi1', b'xi2', b'Mpps', b'Ypps',
               b'Mww/Mpps', b'Ypps-Yww']


#  Função para abertura dos arquivos:

def open_file(file):  # abertura dos arquivos de sinal.
    df = None
    with h5py.File(file, 'r') as f:
        # O data set é um arquivo que pode conter centenas ou até milhares de dados sobre um determinado assunto.
        dset_columns = f['columns']
        dset_dados = f['dados']
        # print('\n colunas  -->', np.array( dset_columns),'\n')
        df = pd.DataFrame(np.array(dset_dados), columns=np.array(dset_columns))
        df[b'Mpps'] = raiz_s * (np.sqrt(df[b'xi1'] * df[b'xi2']))  # Massa  PPS ou Massa perdida.
        df[b'Ypps'] = 1 / 2 * np.log(df[b'xi1'] / df[b'xi2'])  # Pseudorapidez medida no sistema central.
        df[b'Mww/Mpps'] = df[b'W_Mass'] / df[b'Mpps']
        df[b'Ypps-Yww'] = df[b'Ypps'] - df[b'W_rapidity']

        df_cut = (df[b'muon_pt'] > 53) & (df[b'xi1'] > 0.04) & (df[b'xi2'] > 0.04) & (df[b'xi1'] < 0.111) & (
                df[b'xi2'] < 0.138) & (df[b'muon_eta'] < 2.4) & (df[b'jetAK8_pt'] > 200) & (
                         df[b'jetAK8_eta'] < 2.4) & (df[b'METPt'] > 40) & (df[b'W_pt_lep'] > 200)
        dset = df[df_cut]
        return dset


def open_file_2(file):  # abertura dos arquivos de Background.
    df = None
    with h5py.File(file, 'r') as f:
        # O data set é um arquivo que pode conter centenas ou até milhares de dados sobre um determinado assunto.
        dset_columns = f['columns']
        dset_dados = f['dados']
        # print( '\n colunas --> ', np.array( dset_columns ),'\n' )
        df = pd.DataFrame(np.array(dset_dados), columns=np.array(dset_columns))
        df[b'Mpps'] = raiz_s * (np.sqrt(df[b'xi1'] * df[b'xi2']))  # Massa  PPS ou Massa perdida.
        df[b'Ypps'] = 1 / 2 * np.log(df[b'xi1'] / df[b'xi2'])  # Pseudorapidez medida no sistema central.
        df[b'Mww/Mpps'] = df[b'W_Mass'] / df[b'Mpps']
        df[b'Ypps-Yww'] = df[b'Ypps'] - df[b'W_rapidity']

        df_cut = (df[b'muon_pt'] > 53) & (df[b'xi1'] > 0.04) & (df[b'xi2'] > 0.04) & (df[b'xi1'] < 0.111) & (
                df[b'xi2'] < 0.138) & (df[b'muon_eta'] < 2.4) & (df[b'jetAK8_pt'] > 200) & (
                         df[b'jet_eta'] < 2.4) & (df[b'METPt'] > 40) & (df[b'W_pt_lep'] > 200)
        dset = df[df_cut]
        return dset


# Abrindo os arquivos:

# Abrindo e corrigindo os arquivos de background.

TTbar = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_TTbar.h5')), columns=coluna)
DrellYan = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_DrellYan.h5')), columns=coluna)
QCD = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_QCD.h5')), columns=coluna)
sing_top = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_single_top.h5')), columns=coluna)
VV_inclusivo = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_VV_inclusivo.h5')),
                            columns=coluna)
W_jets = pd.DataFrame(np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_WJets.h5')), columns=coluna)

# Abrindo os arquivos de sinal.
#     alhpac = 1 ao 4
#     alpha0 = 5 ao 8

for i in range(1, 9):
    caminho_arquivo = PATH + f'dados/anomalos+sm/output-DataSet_ANOMALO{i}_multiRP.h5'
    nome_variavel = f'anomalo_{i}'
    arquivo_aberto = open_file(caminho_arquivo)
    globals()[nome_variavel] = arquivo_aberto

# Abrindo o arquivo de SM.

SM = open_file(PATH + 'dados/anomalos+sm/output-SM_multiRP.h5')

# Colocando as colunas de rótulo nos arquivos:


# Sinais anomalos.

# Loop para atribuir o valor '1' à chave 'label' para cada variável anomalo_i
for i in range(1, 9):
    nome_variavel = f'anomalo_{i}'
    globals()[nome_variavel]['label'] = 1

# Sinal do SM.

SM['label'] = 1

# Background.

TTbar['label'] = 0
DrellYan['label'] = 0
QCD['label'] = 0
sing_top['label'] = 0
VV_inclusivo['label'] = 0
W_jets['label'] = 0

# Concatenando:

# *Ou* seja juntando as informações de background com os anomalos e o SM. E por fim embaralhando os dados
# concatenados.

# Loop para criar os dataframes x_anomalo_i
for i in range(1, 9):
    nome_variavel = f'x_anomalo_{i}'
    anomalo = globals()[f'anomalo_{i}']
    globals()[nome_variavel] = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo, SM], axis=0))

# Rotulando amostras e criando os conjuntos x e y

# **Conjunto** x:

# --> Criando os x_train e os x_test:
# Loop para dividir os dataframes x_anomalo_i em conjuntos de treino e teste
for i in range(1, 9):
    nome_variavel_train = f'x_anomalo_{i}_train'
    nome_variavel_test = f'x_anomalo_{i}_test'
    x_anomalo_i = globals()[f'x_anomalo_{i}']
    x_anomalo_i_train, x_anomalo_i_test = train_test_split(x_anomalo_i, test_size=teste_size, random_state=42, stratify=x_anomalo_i['label'])
    globals()[nome_variavel_train] = x_anomalo_i_train
    globals()[nome_variavel_test] = x_anomalo_i_test

# --> Determinando os Pesos:

# Loop para criar as variáveis weight_anomalo_i
for i in range(1, 9):
    nome_variavel_weight = f'weight_anomalo_{i}'
    x_anomalo_test = globals()[f'x_anomalo_{i}_test']
    globals()[nome_variavel_weight] = x_anomalo_test[b'weight']

# **Conjunto** y:

# --> Criando os y_train e os y_test:

# Loop para criar as variáveis y_anomalo_i
for i in range(1, 9):
    nome_variavel_y = f'y_anomalo_{i}'
    x_anomalo = globals()[f'x_anomalo_{i}']
    globals()[nome_variavel_y] = x_anomalo['label']

# Loop para criar as variáveis y_anomalo_i_train e y_anomalo_i_test
for i in range(1, 9):
    nome_variavel_y_train = f'y_anomalo_{i}_train'
    nome_variavel_y_test = f'y_anomalo_{i}_test'
    x_anomalo_train = globals()[f'x_anomalo_{i}_train']
    x_anomalo_test = globals()[f'x_anomalo_{i}_test']
    globals()[nome_variavel_y_train] = x_anomalo_train['label']
    globals()[nome_variavel_y_test] = x_anomalo_test['label']

# Loop para selecionar apenas as colunas desejadas para os dataframes x_anomalo_i, x_anomalo_i_train e x_anomalo_i_test
for i in range(1, 9):
    nome_variavel_x = f'x_anomalo_{i}'
    nome_variavel_x_train = f'x_anomalo_{i}_train'
    nome_variavel_x_test = f'x_anomalo_{i}_test'

    x_anomalo = globals()[nome_variavel_x]
    x_anomalo_train = globals()[nome_variavel_x_train]
    x_anomalo_test = globals()[nome_variavel_x_test]

    globals()[nome_variavel_x] = x_anomalo[coluna_LGBM]
    globals()[nome_variavel_x_train] = x_anomalo_train[coluna_LGBM]
    globals()[nome_variavel_x_test] = x_anomalo_test[coluna_LGBM]

# Salvando os x_train, x_test, y_train e y_test:

# Loop para salvar os dados em arquivos .h5
for i in range(1, 9):
    # Criando o nome das variáveis  y_anomalo_{i}_train
    nome_variavel_x_treino = f'x_anomalo_{i}_train'
    nome_variavel_x_teste = f'x_anomalo_{i}_test'
    nome_variavel_y_treino = f'y_anomalo_{i}_train'
    nome_variavel_y_teste = f'y_anomalo_{i}_test'
    nome_variavel_peso = f'weight_anomalo_{i}'

    # Salvando os arquivos de treino
    if not os.path.exists(f'{PATH}{PATH2}v2/x_train/'):
        os.makedirs(f'{PATH}{PATH2}v2/x_train/')
    with h5py.File(PATH + PATH2 + f'v2/x_train/x_anomalo_{i}_train.h5', 'w') as f:
        dset = f.create_dataset('treino', data=globals()[nome_variavel_x_treino])

    if not os.path.exists(f'{PATH}{PATH2}v2/y_train/'):
        os.makedirs(f'{PATH}{PATH2}v2/y_train/')
    with h5py.File(PATH + PATH2 + f'v2/y_train/y_anomalo_{i}_train.h5', 'w') as f:
        dset = f.create_dataset('treino', data=globals()[nome_variavel_y_treino])

    # Salvando os arquivos de teste
    if not os.path.exists(f'{PATH}{PATH2}v2/x_test/'):
        os.makedirs(f'{PATH}{PATH2}v2/x_test/')
    with h5py.File(PATH + PATH2 + f'v2/x_test/x_anomalo_{i}_test.h5', 'w') as f:
        dset = f.create_dataset('teste', data=globals()[nome_variavel_x_teste])

    if not os.path.exists(f'{PATH}{PATH2}v2/y_test/'):
        os.makedirs(f'{PATH}{PATH2}v2/y_test/')
    with h5py.File(PATH + PATH2 + f'v2/y_test/y_anomalo_{i}_test.h5', 'w') as f:
        dset = f.create_dataset('teste', data=globals()[nome_variavel_y_teste])

    # Salvando os arquivos de peso
    if not os.path.exists(f'{PATH}{PATH2}v2/weight_anomalo/'):
        os.makedirs(f'{PATH}{PATH2}v2/weight_anomalo/')
    with h5py.File(PATH + PATH2 + f'v2/weight_anomalo/weight_anomalo_{i}.h5', 'w') as f:
        dset = f.create_dataset('peso', data=globals()[nome_variavel_peso])

