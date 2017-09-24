#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:44:22 2017

@author: amara
"""
#importando as bibliotecas necessárias
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import table
from sklearn import preprocessing
#carregando o banco de dados como um DataFrame do pandas
dados = pd.read_csv("dados-07.csv")
#Descrevendo e verificando tipos dos atributos
desc_dados=dados.describe()
types_dados=dados.dtypes
print(desc_dados)
print(types_dados)
#Gerando tabelas com descrições
ax = plt.subplot(111, frame_on=False) # sem grades visiveis
ax.xaxis.set_visible(False)  # oculta eixo x
ax.yaxis.set_visible(False)  # oculta eixo y
table(ax, desc_dados)  # onde desc e types sao os dataframes
plt.savefig('desc.png')

ax = plt.subplot(111, frame_on=False) # sem grades visiveis
ax.xaxis.set_visible(False)  # oculta eixo x
ax.yaxis.set_visible(False)  # oculta eixo y
table(ax, types_dados)  # onde desc e types sao os dataframes
plt.savefig('types.png')
#Verificando dados faltantes
desc_dados2=dados.dropna().describe()
ax = plt.subplot(111, frame_on=False) # sem grades visiveis
ax.xaxis.set_visible(False)  # oculta eixo x
ax.yaxis.set_visible(False)  # oculta eixo y
table(ax, desc_dados2)  # onde desc e types sao os dataframes
plt.savefig('dados2.png')
#Criando DataFrame ignorando dados faltantes
dados2=dados.dropna()

#Convertendo a coluna "avaliacao-usuarios" para valores numericos
dados2['avaliacao-usuarios'] = dados2['avaliacao-usuarios'].convert_objects(convert_numeric=True)
#Normalizando dados numericos
#scaler = preprocessing.MinMaxScaler()
#dados2[["vendas"]]=pd.DataFrame(scaler.fit_transform(dados2[["vendas"]]))
#dados2[["numero-usuarios"]]=pd.DataFrame(scaler.fit_transform(dados2[["numero-usuarios"]]))
#dados2[["numero-criticos"]]=pd.DataFrame(scaler.fit_transform(dados2[["numero-criticos"]]))
#dados2[["avaliacao-usuarios"]]=pd.DataFrame(scaler.fit_transform(dados2[["avaliacao-usuarios"]]))
#dados2[["avaliacao-criticos"]]=pd.DataFrame(scaler.fit_transform(dados2[["avaliacao-criticos"]]))


dados2[["vendas"]]=pd.DataFrame(preprocessing.scale(dados2[["vendas"]]))
dados2[["numero-usuarios"]]=pd.DataFrame(preprocessing.scale(dados2[["numero-usuarios"]]))
dados2[["numero-criticos"]]=pd.DataFrame(preprocessing.scale(dados2[["numero-criticos"]]))
dados2[["avaliacao-usuarios"]]=pd.DataFrame(preprocessing.scale(dados2[["avaliacao-usuarios"]]))
dados2[["avaliacao-criticos"]]=pd.DataFrame(preprocessing.scale(dados2[["avaliacao-criticos"]]))










