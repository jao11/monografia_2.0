# Instalando bibliotecas e fazendo imports necessarios:

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


import h5py
import numpy as np
import pandas as pd

# Constantes facilitadoras e colunas:

raiz_s = 13000 # GeV - consante que representa a energia na colisão dos prótons
teste_size = 0.3 # ??
PATH = '/home/professora/Documentos/material_uerj/monografia/monografia/'
PATH2 = 'codigo1/'

coluna = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta', b'jetAK8_prunedMass', b'jetAK8_tau21',
            b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'PUWeight', b'W_rapidity', b'btag', b'xi1', b'xi2', b'ismultirp1', b'ismultirp2',
            b'Norm', b'weight', b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww']

coluna_LGBM = [b'W_Mass', b'W_pt_lep', b'dPhi_Whad_Wlep', b'dPhi_jatos_MET', b'jetAK8_pt', b'jetAK8_eta', b'jetAK8_prunedMass', b'jetAK8_tau21',
            b'METPt', b'muon_pt', b'muon_eta', b'ExtraTracks', b'W_rapidity', b'xi1', b'xi2',  b'Mpps', b'Ypps', b'Mww/Mpps', b'Ypps-Yww']

#  Função para abertura dos arquivos:

def open_file( file ): # abertura dos arquivos de sinal.
  df = None
  with h5py.File(file, 'r') as f:
    # O data set é um arquivo que pode conter centenas ou até milhares de dados sobre um determinado assunto.
    dset_columns = f['columns']
    dset_dados = f['dados']
    # print('\n colunas  -->', np.array( dset_columns),'\n')
    df = pd.DataFrame( np.array(dset_dados), columns = np.array(dset_columns))
    df[b'Mpps'] = raiz_s * ( np.sqrt( df[b'xi1'] * df[b'xi2'] ) ) # Massa  PPS ou Massa perdida.
    df[b'Ypps'] = 1/2 * np.log( df[b'xi1'] / df[b'xi2'] ) # Pseudorapidez medida no sistema central.
    df[b'Mww/Mpps'] = df[b'W_Mass'] / df[b'Mpps']
    df[b'Ypps-Yww'] = df[b'Ypps'] - df[b'W_rapidity']

    df_cut = (df[b'muon_pt'] > 53) & (df[b'xi1'] > 0.04) & (df[b'xi2'] > 0.04) & (df[b'xi1'] < 0.111) & (df[b'xi2'] < 0.138) & (df[b'muon_eta'] < 2.4) & (df[b'jetAK8_pt'] > 200) & (df[b'jetAK8_eta'] < 2.4) & (df[b'METPt'] > 40) & (df[b'W_pt_lep'] > 200)
    dset = df[df_cut]
    return dset

def open_file_2( file ): # abertura dos arquivos de Background.
  df = None
  with h5py.File( file , 'r' ) as f:
    # O data set é um arquivo que pode conter centenas ou até milhares de dados sobre um determinado assunto.
    dset_columns = f['columns']
    dset_dados = f['dados']
    # print( '\n colunas --> ', np.array( dset_columns ),'\n' )
    df = pd.DataFrame( np.array(dset_dados), columns = np.array( dset_columns))
    df[b'Mpps'] = raiz_s * ( np.sqrt( df[b'xi1'] * df[b'xi2'] ) ) # Massa  PPS ou Massa perdida.
    df[b'Ypps'] = 1/2 * np.log( df[b'xi1'] / df[b'xi2'] ) # Pseudorapidez medida no sistema central.
    df[b'Mww/Mpps'] = df[b'W_Mass'] / df[b'Mpps']
    df[b'Ypps-Yww'] = df[b'Ypps'] - df[b'W_rapidity']

    df_cut = (df[b'muon_pt'] > 53) & (df[b'xi1'] > 0.04) & (df[b'xi2'] > 0.04) & (df[b'xi1'] < 0.111) & (df[b'xi2'] < 0.138) & (df[b'muon_eta'] < 2.4) &   (df[b'jetAK8_pt'] > 200) & (df[b'jet_eta'] < 2.4) & (df[b'METPt'] > 40) & (df[b'W_pt_lep'] > 200)
    dset = df[df_cut]
    return dset

# Abrindo os arquivos:

# Abrindo e corrigindo os arquivos de background.

TTbar = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_TTbar.h5')), columns = coluna)
DrellYan = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_DrellYan.h5')),columns = coluna)
QCD = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_QCD.h5')),columns = coluna)
sing_top = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_single_top.h5')),columns = coluna)
VV_inclusivo = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_VV_inclusivo.h5')),columns = coluna)
W_jets = pd.DataFrame( np.array(open_file_2(PATH + 'dados/background/DataSet_multiRP_WJets.h5')),columns = coluna)

# Abrindo os arquivos de sinal.
#     alhpac = 1 ao 4
#     alpha0 = 5 ao 8


anomalo_1 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO1_multiRP.h5')
anomalo_2 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO2_multiRP.h5')
anomalo_3 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO3_multiRP.h5')
anomalo_4 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO4_multiRP.h5')
anomalo_5 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO5_multiRP.h5')
anomalo_6 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO6_multiRP.h5')
anomalo_7 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO7_multiRP.h5')
anomalo_8 = open_file(PATH + 'dados/anomalos+sm/output-DataSet_ANOMALO8_multiRP.h5')

# Abrindo o arquivo de SM.

SM = open_file(PATH + 'dados/anomalos+sm/output-SM_multiRP.h5')

# Colocando as colunas de rótulo nos arquivos:


# Sinais anomalos.

anomalo_1['label'] = 1
anomalo_2['label'] = 1
anomalo_3['label'] = 1
anomalo_4['label'] = 1
anomalo_5['label'] = 1
anomalo_6['label'] = 1
anomalo_7['label'] = 1
anomalo_8['label'] = 1

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

x_anomalo_1 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_1, SM], axis=0))
x_anomalo_2 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_2, SM], axis=0))
x_anomalo_3 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_3, SM], axis=0))
x_anomalo_4 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_4, SM], axis=0))
x_anomalo_5 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_5, SM], axis=0))
x_anomalo_6 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_6, SM], axis=0))
x_anomalo_7 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_7, SM], axis=0))
x_anomalo_8 = shuffle(pd.concat([TTbar, DrellYan, QCD, VV_inclusivo, W_jets, sing_top, anomalo_8, SM], axis=0))

# Rotulando amostras e criando os conjuntos x e y

# **Conjunto** x:

# --> Criando os x_train e os x_test:

x_anomalo_1_train, x_anomalo_1_test = train_test_split( x_anomalo_1, test_size = teste_size, random_state = 42, stratify = x_anomalo_1.label )
x_anomalo_2_train, x_anomalo_2_test = train_test_split( x_anomalo_2, test_size = teste_size, random_state = 42, stratify = x_anomalo_2.label )
x_anomalo_3_train, x_anomalo_3_test = train_test_split( x_anomalo_3, test_size = teste_size, random_state = 42, stratify = x_anomalo_3.label )
x_anomalo_4_train, x_anomalo_4_test = train_test_split( x_anomalo_4, test_size = teste_size, random_state = 42, stratify = x_anomalo_4.label )
x_anomalo_5_train, x_anomalo_5_test = train_test_split( x_anomalo_5, test_size = teste_size, random_state = 42, stratify = x_anomalo_5.label )
x_anomalo_6_train, x_anomalo_6_test = train_test_split( x_anomalo_6, test_size = teste_size, random_state = 42, stratify = x_anomalo_6.label )
x_anomalo_7_train, x_anomalo_7_test = train_test_split( x_anomalo_7, test_size = teste_size, random_state = 42, stratify = x_anomalo_7.label )
x_anomalo_8_train, x_anomalo_8_test = train_test_split( x_anomalo_8, test_size = teste_size, random_state = 42, stratify = x_anomalo_8.label )

# --> Determinando os Pesos:

weight_anomalo_1 = x_anomalo_1_test[b'weight']
weight_anomalo_2 = x_anomalo_2_test[b'weight']
weight_anomalo_3 = x_anomalo_3_test[b'weight']
weight_anomalo_4 = x_anomalo_4_test[b'weight']
weight_anomalo_5 = x_anomalo_5_test[b'weight']
weight_anomalo_6 = x_anomalo_6_test[b'weight']
weight_anomalo_7 = x_anomalo_7_test[b'weight']
weight_anomalo_8 = x_anomalo_8_test[b'weight']

# **Conjunto** y:

# --> Criando os y_train e os y_test:

y_anomalo_1= x_anomalo_1['label']
y_anomalo_2= x_anomalo_2['label']
y_anomalo_3= x_anomalo_3['label']
y_anomalo_4= x_anomalo_4['label']
y_anomalo_5= x_anomalo_5['label']
y_anomalo_6= x_anomalo_6['label']
y_anomalo_7= x_anomalo_7['label']
y_anomalo_8= x_anomalo_8['label']


y_anomalo_1_train = x_anomalo_1_train['label']
y_anomalo_1_test  = x_anomalo_1_test['label']

y_anomalo_2_train = x_anomalo_2_train['label']
y_anomalo_2_test  = x_anomalo_2_test['label']

y_anomalo_3_train = x_anomalo_3_train['label']
y_anomalo_3_test  = x_anomalo_3_test['label']

y_anomalo_4_train = x_anomalo_4_train['label']
y_anomalo_4_test  = x_anomalo_4_test['label']

y_anomalo_5_train = x_anomalo_5_train['label']
y_anomalo_5_test  = x_anomalo_5_test['label']

y_anomalo_6_train = x_anomalo_6_train['label']
y_anomalo_6_test  = x_anomalo_6_test['label']

y_anomalo_7_train = x_anomalo_7_train['label']
y_anomalo_7_test  = x_anomalo_7_test['label']

y_anomalo_8_train = x_anomalo_8_train['label']
y_anomalo_8_test  = x_anomalo_8_test['label']


# *Conjunto* x com coluna LGBM:

x_anomalo_1 = x_anomalo_1[coluna_LGBM]
x_anomalo_1_train = x_anomalo_1_train[coluna_LGBM]
x_anomalo_1_test  = x_anomalo_1_test[coluna_LGBM]

x_anomalo_2 = x_anomalo_2[coluna_LGBM]
x_anomalo_2_train = x_anomalo_2_train[coluna_LGBM]
x_anomalo_2_test  = x_anomalo_2_test[coluna_LGBM]

x_anomalo_3 = x_anomalo_3[coluna_LGBM]
x_anomalo_3_train = x_anomalo_3_train[coluna_LGBM]
x_anomalo_3_test  = x_anomalo_3_test[coluna_LGBM]

x_anomalo_4 = x_anomalo_4[coluna_LGBM]
x_anomalo_4_train = x_anomalo_4_train[coluna_LGBM]
x_anomalo_4_test  = x_anomalo_4_test[coluna_LGBM]

x_anomalo_5 = x_anomalo_5[coluna_LGBM]
x_anomalo_5_train = x_anomalo_5_train[coluna_LGBM]
x_anomalo_5_test  = x_anomalo_5_test[coluna_LGBM]

x_anomalo_6 = x_anomalo_6[coluna_LGBM]
x_anomalo_6_train = x_anomalo_6_train[coluna_LGBM]
x_anomalo_6_test  = x_anomalo_6_test[coluna_LGBM]

x_anomalo_7 = x_anomalo_7[coluna_LGBM]
x_anomalo_7_train = x_anomalo_7_train[coluna_LGBM]
x_anomalo_7_test  = x_anomalo_7_test[coluna_LGBM]

x_anomalo_8 = x_anomalo_8[coluna_LGBM]
x_anomalo_8_train = x_anomalo_8_train[coluna_LGBM]
x_anomalo_8_test  = x_anomalo_8_test[coluna_LGBM]


# Salvando os x_train, x_test, y_train e y_test:

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_1_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_1_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_1_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_1_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_1_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_1_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_1_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_1_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_1.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_1)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_2_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_2_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_2_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_2_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_2_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_2_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_2_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_2_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_2.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_2)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_3_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_3_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_3_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_3_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_3_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_3_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_3_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_3_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_3.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_3)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_4_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_4_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_4_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_4_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_4_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_4_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_4_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_4_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_4.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_4)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_5_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_5_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_5_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_5_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_5_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_5_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_5_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_5_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_5.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_5)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_6_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_6_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_6_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_6_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_6_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_6_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_6_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_6_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_6.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_6)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_7_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_7_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_7_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_7_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_7_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_7_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_7_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_7_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_7.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_7)

with h5py.File(PATH + PATH2 + 'x_train/x_anomalo_8_train.h5', 'w') as f:
    dset = f.create_dataset("treino", data=x_anomalo_8_train)
    with h5py.File(PATH + PATH2 + 'x_test/x_anomalo_8_test.h5', 'w') as f:
        dset = f.create_dataset("treino", data=x_anomalo_8_test)
        with h5py.File(PATH + PATH2 + 'y_train/y_anomalo_8_train.h5', 'w') as f:
            dset = f.create_dataset("treino", data=y_anomalo_8_train)
            with h5py.File(PATH + PATH2 + 'y_test/y_anomalo_8_test.h5', 'w') as f:
                dset = f.create_dataset("treino", data=y_anomalo_8_test)
                with h5py.File(PATH + PATH2 + 'weight_anomalo/weight_anomalo_8.h5', 'w') as f:
                    dset = f.create_dataset("treino", data=weight_anomalo_8)








