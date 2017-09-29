""""
Em relação ao arquivo trabalho_01.py, seguem as alterações:
    1 - Remoção dos outliers do arquivo original antes de preencher os valores ausentes com a média por atributo.
        O que torna o DataFrame 'dados' do escopo trabalho_01.py DIFERENTE do DataFrame 'dados' deste escopo.
    
    2 - Substituição das ocorrências 'TBA' e 'Canceled' no atributo 'lancamento' por 'nan'
    
    
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

# Configurando avisos do Pandas
pd.options.mode.chained_assignment = None

#Parte referente ao Trabalho-01

# Carregando os dados e ajustando os tipos
dados = pd.read_csv("dados-07.csv",
                    dtype={ 
                            "nome":"category",
                            "plataforma":"category",
                            "genero":"category",
                            "editora":"category",
                            "lancamento":"category",
                            "fabricante":"category",
                            "vendas":"float",
                            "avaliacao-criticos":"float",
                            "numero-criticos":"float",
                            "numero-usuarios":"float"
                           }
                    )

# Ajustando inconsistência e tipo no atributo 'avaliacao-usuarios':
# Substituição dos valores valores 'tbd' por 'nan' no atributo e conversão para float
# Substituição dos valores valores 'TBA' e 'Canceled' no atributo 'lancamento' por 'nan'
dados["avaliacao-usuarios"]= dados["avaliacao-usuarios"].replace({"tbd":"-999"})
dados["avaliacao-usuarios"] = dados["avaliacao-usuarios"].astype(float)
dados["avaliacao-usuarios"] = dados["avaliacao-usuarios"].replace({-999:math.nan})
#Tratando as inconsistencias do atributo 'lancamento'
dados["lancamento"] = dados["lancamento"].replace({"TBA":"-999"})
dados["lancamento"] = dados["lancamento"].replace({"Canceled":"-999"})
dados["lancamento"] = dados["lancamento"].replace({math.nan:"-999"})
dados["lancamento"] = dados["lancamento"].replace({"-999":"1-Jan-99"})
dados["lancamento"] = dados["lancamento"].replace({"1-06-10":"1-Jun-10"})
dados["lancamento"] = dados["lancamento"].replace({"Apr-10":"1-Apr-10"})
dados["lancamento"] = dados["lancamento"].replace({"Jun-10":"1-Jun-10"})
#Convertendo o atributo 'lancamento' para datetime e selecionando apenas o mes
dados['lancamento'] = pd.to_datetime(dados['lancamento'])
dados['lancamento'] = dados['lancamento'].dt.month
#Preenchendo os dados faltantes em 'fabricante' por "indisponivel' e transformando-os em dados categoricos
dados["fabricante"] = dados["fabricante"].replace({math.nan:"indisponivel"})
dados.fabricante = dados.fabricante.astype('category')



print()
print("============================== ANTES DOS TRATAMENTOS ==============================")
print()

# Imprime os tipos dos dados
print("=> Tipos dos dados:")
dados_types = dados.dtypes
print(dados_types)
print()

# Imprime a descrição dos dados
print("=> Descrição dos dados:")
dados_descr = dados.describe()
print(dados_descr)
print()

# Imprime a quantidade de valores ausentes em cada atributo
print("=> Valores ausentes:")
print("nome: ", dados["nome"].isnull().sum())
print("plataforma: ", dados["plataforma"].isnull().sum())
print("genero: ", dados["genero"].isnull().sum())
print("editora: ", dados["editora"].isnull().sum())
print("lancamento: ", dados["lancamento"].isnull().sum())
print("fabricante: ", dados["fabricante"].isnull().sum())
print("vendas: ", dados["vendas"].isnull().sum())
print("avaliacao-criticos: ", dados["avaliacao-criticos"].isnull().sum())
print("numero-criticos: ", dados["numero-criticos"].isnull().sum())
print("avaliacao-usuarios: ", dados["avaliacao-usuarios"].isnull().sum())
print("numero-usuarios: ", dados["numero-usuarios"].isnull().sum())
print()

# Filtrando os dados por editora, gênero, quantidade de avaliações e avaliações
dados_filter1 = dados[["editora", "genero", "numero-criticos", "avaliacao-criticos", "numero-usuarios", "avaliacao-usuarios"]]

# Limpando os dados que possuem valores ausentes
dados_filter1_clean = dados_filter1.dropna()

# Filtrando os dados_filter_1 por editora, gênero e avaliações dos críticos
dados_filter2 = dados_filter1_clean[["editora", "genero", "avaliacao-criticos"]]

# Média de avaliação dos críticos por editora e gênero
dados_filter2_mean = dados_filter2.groupby(["editora", "genero"])["avaliacao-criticos"].mean()

# Filtrando os dados_filter_1 por editora, gênero e avaliações dos usuários
dados_filter3 = dados_filter1_clean[["editora", "genero", "avaliacao-usuarios"]]

# Média de avaliação dos usuários por editora e gênero
dados_filter3_mean = dados_filter3.groupby(["editora", "genero"])["avaliacao-usuarios"].mean()

print()
print("==============================EXECUTANDO O TRATAMENTO ==============================")
print()

# Identificando outliers:

# vendas
dados_outlier_vendas = dados[["nome", "vendas"]]
dados_filter4 = dados_outlier_vendas["vendas"]

q1, q3 = dados_outlier_vendas["vendas"].quantile([0.25, 0.75])
iqr = (q3-q1)
min = q1 - (iqr * 1.5)
max = q3 + (iqr * 1.5)

dados_outlier_vendas["vendas"] = dados_filter4[(dados_filter4 > min) & (dados_filter4  < max)]
dados_outlier_vendas = dados_outlier_vendas.loc[dados_outlier_vendas["vendas"].isnull()]
dados_outlier_vendas = dados_outlier_vendas["nome"]

# avaliacao-criticos
dados_outlier_avaliacaocriticos = dados[["nome", "avaliacao-criticos"]]
dados_filter4 = dados_outlier_avaliacaocriticos["avaliacao-criticos"]

q1, q3 = dados_outlier_avaliacaocriticos["avaliacao-criticos"].quantile([0.25, 0.75])
iqr = (q3-q1)
min = q1 - (iqr * 1.5)
max = q3 + (iqr * 1.5)

dados_outlier_avaliacaocriticos["avaliacao-criticos"] = dados_filter4[(dados_filter4 > min) & (dados_filter4  < max)]
dados_outlier_avaliacaocriticos = dados_outlier_avaliacaocriticos.loc[dados_outlier_avaliacaocriticos["avaliacao-criticos"].isnull()]
dados_outlier_avaliacaocriticos = dados_outlier_avaliacaocriticos["nome"]

# numero-criticos
dados_outlier_numerocriticos = dados[["nome", "numero-criticos"]]
dados_filter4 = dados_outlier_numerocriticos["numero-criticos"]

q1, q3 = dados_outlier_numerocriticos["numero-criticos"].quantile([0.25, 0.75])
iqr = (q3-q1)
min = q1 - (iqr * 1.5)
max = q3 + (iqr * 1.5)

dados_outlier_numerocriticos["numero-criticos"] = dados_filter4[(dados_filter4 > min) & (dados_filter4  < max)]
dados_outlier_numerocriticos = dados_outlier_numerocriticos.loc[dados_outlier_numerocriticos["numero-criticos"].isnull()]
dados_outlier_numerocriticos = dados_outlier_numerocriticos["nome"]

# avaliacao-usuarios
dados_outlier_avaliacaousuarios = dados[["nome", "avaliacao-usuarios"]]
dados_filter4 = dados_outlier_avaliacaousuarios["avaliacao-usuarios"]

q1, q3 = dados_outlier_avaliacaousuarios["avaliacao-usuarios"].quantile([0.25, 0.75])
iqr = (q3-q1)
min = q1 - (iqr * 1.5)
max = q3 + (iqr * 1.5)

dados_outlier_avaliacaousuarios["avaliacao-usuarios"] = dados_filter4[(dados_filter4 > min) & (dados_filter4  < max)]
dados_outlier_avaliacaousuarios = dados_outlier_avaliacaousuarios.loc[dados_outlier_avaliacaousuarios["avaliacao-usuarios"].isnull()]
dados_outlier_avaliacaousuarios = dados_outlier_avaliacaousuarios["nome"]

# numero-usuarios
dados_outlier_numerousuarios = dados[["nome", "numero-usuarios"]]
dados_filter4 = dados_outlier_numerousuarios["numero-usuarios"]

q1, q3 = dados_outlier_numerousuarios["numero-usuarios"].quantile([0.25, 0.75])
iqr = (q3-q1)
min = q1 - (iqr * 1.5)
max = q3 + (iqr * 1.5)

dados_outlier_numerousuarios["numero-usuarios"] = dados_filter4[(dados_filter4 > min) & (dados_filter4  < max)]
dados_outlier_numerousuarios = dados_outlier_numerousuarios.loc[dados_outlier_numerousuarios["numero-usuarios"].isnull()]
dados_outlier_numerousuarios = dados_outlier_numerousuarios["nome"]

# Faz a substituição dos valores ausentes de avaliação dos críticos pela média de avaliação por editora e gênero
# Ou caso não existam dados para isto, substitui o valor ausente por um valor constante
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if math.isnan(game[3]):
        if ((game[0], game[1]) in dados_filter2_mean):
            game[3] = dados_filter2_mean[{game[0], game[1]}].values[0]
            dados_filter1.set_value(i, "avaliacao-criticos", game[3])
        else:
            dados_filter1.set_value(i, "avaliacao-criticos", 0)

# Faz a substituição dos valores ausentes de avaliação dos usuários pela média de avaliação por editora e gênero
# Ou caso não existam dados para isto, substitui o valor ausente por um valor constante
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if math.isnan(game[5]):
        if ((game[0], game[1]) in dados_filter3_mean):
            game[5] = dados_filter3_mean[{game[0], game[1]}].values[0]
            dados_filter1.set_value(i, "avaliacao-usuarios", game[5])
        else:
            dados_filter1.set_value(i, "avaliacao-usuarios", 0)
        
# Filtrando os dados_filter_1 por editora, gênero e quantidade de críticos
dados_filter2 = dados_filter1_clean[["editora", "genero", "numero-criticos"]]

# Média de avaliação dos críticos por editora e gênero
dados_filter2_mean = dados_filter2.groupby(["editora", "genero"])["numero-criticos"].mean()

# Filtrando os dados_filter_1 por editora, gênero e quantidade de usuários
dados_filter3 = dados_filter1_clean[["editora", "genero", "numero-usuarios"]]

# Média de avaliação dos usuários por editora e gênero
dados_filter3_mean = dados_filter3.groupby(["editora", "genero"])["numero-usuarios"].mean()
        
# Faz a substituição dos valores ausentes de avaliação dos críticos pela média de avaliação por editora e gênero
# Ou caso não existam dados para isto, substitui o valor ausente por um valor constante
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if math.isnan(game[2]):
        if ((game[0], game[1]) in dados_filter2_mean):
            game[2] = dados_filter2_mean[{game[0], game[1]}].values[0]
            dados_filter1.set_value(i, "numero-criticos", game[2])
        else:
            dados_filter1.set_value(i, "numero-criticos", 0)

# Faz a substituição dos valores ausentes de avaliação dos usuários pela média de avaliação por editora e gênero
# Ou caso não existam dados para isto, substitui o valor ausente por um valor constante
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if math.isnan(game[4]):
        if ((game[0], game[1]) in dados_filter3_mean):
            game[4] = dados_filter3_mean[{game[0], game[1]}].values[0]
            dados_filter1.set_value(i, "numero-usuarios", game[4])
        else:
            dados_filter1.set_value(i, "numero-usuarios", 0)

# Ajusta os dados originais
dados["numero-criticos"] = dados_filter1["numero-criticos"]
dados["avaliacao-criticos"] = dados_filter1["avaliacao-criticos"]
dados["numero-usuarios"] = dados_filter1["numero-usuarios"]
dados["avaliacao-usuarios"] = dados_filter1["avaliacao-usuarios"]


print()
print("============================== APÓS OS TRATAMENTOS ==============================")
print()

# Imprime a descrição dos dados após o tratamento
print("=> Descrição dos dados:")
dados_descr = dados.describe()
print(dados_descr)
print()

# Imprime a quantidade de valores ausentes em cada atributo depois do tratamento
print("=> Valores ausentes:")
print("nome: ", dados["nome"].isnull().sum())
print("plataforma: ", dados["plataforma"].isnull().sum())
print("genero: ", dados["genero"].isnull().sum())
print("editora: ", dados["editora"].isnull().sum())
print("lancamento: ", dados["lancamento"].isnull().sum())
print("fabricante: ", dados["fabricante"].isnull().sum())
print("vendas: ", dados["vendas"].isnull().sum())
print("avaliacao-criticos: ", dados["avaliacao-criticos"].isnull().sum())
print("numero-criticos: ", dados["numero-criticos"].isnull().sum())
print("avaliacao-usuarios: ", dados["avaliacao-usuarios"].isnull().sum())
print("numero-usuarios: ", dados["numero-usuarios"].isnull().sum())
print()

# Imprime a quantidade de outliers
print("=> Quantidade de outliers:")
print("vendas: ", len(dados_outlier_vendas))
print("avaliacao-criticos: ", len(dados_outlier_avaliacaocriticos))
print("numero-criticos: ", len(dados_outlier_numerocriticos))
print("avaliacao-usuarios: ", len(dados_outlier_avaliacaousuarios))
print("numero-usuarios: ", len(dados_outlier_numerousuarios))
print()

# Gera histograma com a distribuição de vendas dos jogos
plt.hist(dados["vendas"])
plt.show()

# Gera gráfico de barras das vendas por plataforma
dados[["plataforma", "vendas"]].groupby(["plataforma"]).sum().plot.bar()
plt.show()

# Gera gráfico de barras das vendas por genero
dados[["genero", "vendas"]].groupby(["genero"]).sum().plot.bar()
plt.show()

# Gera gráfico de barras para a quantidade de jogos de tiro por plataforma
dados_shooter = dados.loc[dados["genero"] == "Shooter"]
dados_shooter[["plataforma", "genero"]].groupby(["plataforma"]).count().plot.bar()
plt.show()

# Gera gráfico de barras para a quantidade de jogos de ação por plataforma
dados_action = dados.loc[dados["genero"] == "Action"]
dados_action[["plataforma", "genero"]].groupby(["plataforma"]).count().plot.bar()
plt.show()

# Gera gráfico de barras para a quantidade de jogos por genero da editora Activision
dados_activision = dados.loc[dados["editora"] == "Activision"]
dados_activision[["editora", "genero"]].groupby(["genero"]).count().plot.bar()
plt.show()

# Gera gráfico de barras para a quantidade de jogos por plataforma da editora Activision
dados_activision = dados.loc[dados["editora"] == "Activision"]
dados_activision[["editora", "plataforma"]].groupby(["plataforma"]).count().plot.bar()
plt.show()

# Gera boxplot dos atributos numéricos
plt.boxplot(dados["vendas"])
plt.show()

plt.boxplot(dados["avaliacao-criticos"])
plt.show()

plt.boxplot(dados["numero-criticos"])
plt.show()

plt.boxplot(dados["avaliacao-usuarios"])
plt.show()

plt.boxplot(dados["numero-usuarios"])
plt.show()

# Liberando variáveis temporárias
del dados_filter1, dados_filter1_clean, dados_filter2, dados_filter2_mean, dados_filter3, dados_filter3_mean, game, i
del dados_filter4, q1, q3, iqr, min, max

#Parte referente ao Trabalho-02

#Criando um arquivo de dados alternativo 2 com base no original excluindo todas as instâncias com pelo menos
#um atributo ausente
dados2 = pd.read_csv("dados-07.csv",
                    dtype={ 
                            "nome":"category",
                            "plataforma":"category",
                            "genero":"category",
                            "editora":"category",
                            "lancamento":"category",
                            "fabricante":"category",
                            "vendas":"float",
                            "avaliacao-criticos":"float",
                            "numero-criticos":"float",
                            "numero-usuarios":"float"
                           }
                    )

dados2=dados2.dropna()
#Convertendo o atributo "avaliacao-usuarios" para tipo float
dados2['avaliacao-usuarios']=pd.to_numeric(dados2['avaliacao-usuarios'])
#Convertendo o atributo 'lancamento' para datetime e selecionando apenas o mes
dados2['lancamento'] = pd.to_datetime(dados2['lancamento'])
dados2['lancamento'] = dados2['lancamento'].dt.month

##########################################################################################################
#Excluindo o atributo 'nome' do dataframe 'dados'
del dados["nome"]
#selecionando as atributos categoricos de 'dados'
dados_categoricos = dados.select_dtypes(['category']).columns
#Transforma os dados categoricos de 'dados' de acordo com os indices gerados
dados[dados_categoricos] = dados[dados_categoricos].apply(lambda x: x.cat.codes)