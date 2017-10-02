import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics

# Configurando avisos do Pandas
pd.options.mode.chained_assignment = None

# Carregando os dados e ajustando os tipos
dados = pd.read_csv("dados-07.csv",
                    dtype={ 
                            "nome":"category",
                            "plataforma":"category",
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
dados["avaliacao-usuarios"]= dados["avaliacao-usuarios"].replace({"tbd":"-999"})
dados["avaliacao-usuarios"] = dados["avaliacao-usuarios"].astype(float)
dados["avaliacao-usuarios"] = dados["avaliacao-usuarios"].replace({-999:math.nan})

dados["lancamento"] = dados["lancamento"].replace({"TBA":math.nan})
dados["lancamento"] = dados["lancamento"].replace({"Cancelled":math.nan})

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

# Liberando variáveis temporárias
del dados_filter1, dados_filter1_clean, dados_filter2, dados_filter2_mean, dados_filter3, dados_filter3_mean, game, i
del dados_filter4, q1, q3, iqr, min, max

#============================== Parte referente ao Trabalho-02 ==============================#

# Removendo outliers
dados_without_outlier = dados.drop(dados_outlier_vendas.index)
dados_without_outlier = dados.drop(dados_outlier_avaliacaocriticos.index)
dados_without_outlier = dados.drop(dados_outlier_numerocriticos.index)
dados_without_outlier = dados.drop(dados_outlier_avaliacaousuarios.index)
dados_without_outlier = dados.drop(dados_outlier_numerousuarios.index)

# Removendo nulos se houver
dados_clean = dados_without_outlier.dropna()

# Removendo atributos para o agrupamento
del dados_clean["nome"], dados_clean["plataforma"], dados_clean["genero"], dados_clean["lancamento"]

# Selecionando as atributos categóricos
dados_clean_category = dados_clean.select_dtypes(["category"]).columns

# Transforma os dados categóricos para numéricos
dados_clean[dados_clean_category] = dados[dados_clean_category].apply(lambda x: x.cat.codes)

# Normalizando os dados
scaler = preprocessing.MinMaxScaler()
dados_clean_norm = scaler.fit_transform(dados_clean)

# Agrupamento utilizando o kmeans
clusters = 8
kmeans = cluster.KMeans(n_clusters=clusters)
dados_clean_kmeans = kmeans.fit_predict(dados_clean_norm)

# Avaliando o agrupamento com o índice de silhueta
print("KMeans-Silueta: ", metrics.silhouette_score(dados_clean_norm, dados_clean_kmeans, metric="euclidean"))

# Recuperando as informações em um Dataframe
dados_clean_kmeans = pd.DataFrame(dados_clean_kmeans, columns=["group"])

dt1 = dados_without_outlier.dropna()
dt1 = dt1[["nome", "plataforma", "genero"]]

dt1 = dt1.reset_index(drop=True)
dados_clean = dados_clean.reset_index(drop=True)
dados_clean_kmeans = dados_clean_kmeans.reset_index(drop=True)

new_dataframe = [dt1, dados_clean, dados_clean_kmeans]

# Dados agrupados
dados_groups = pd.concat(new_dataframe, axis=1)

# Gera um gráfico com a quantidade de jogos por plataforma para cada grupo
for i in range(0, clusters):
    dados_group_i = dados_groups.loc[dados_groups["group"] == i]
    dados_group_i[["plataforma", "nome"]].groupby(["plataforma"]).count().plot.bar()
    plt.show()
    
# Gera um gráfico com a quantidade de jogos de cada gênero para cada grupo
for i in range(0, clusters):
    dados_group_i = dados_groups.loc[dados_groups["group"] == i]
    dados_group_i[["genero", "nome"]].groupby(["genero"]).count().plot.bar()
    plt.show()

# Gera um histograma com as vendas dos jogos de cada grupo
for i in range(0, clusters):
    dados_group_i = dados_groups.loc[dados_groups["group"] == i]
    plt.hist(dados_group_i["vendas"])
    plt.show()

# Gera um gráfico com as vendas por plataforma para cada grupo
for i in range(0, clusters):
    dados_group_i = dados_groups.loc[dados_groups["group"] == i]
    dados_group_i[["plataforma", "vendas"]].groupby(["plataforma"]).sum().plot.bar()
    plt.show()

# Gera um gráfico com as vendas por gênero para cada grupo
for i in range(0, clusters):
    dados_group_i = dados_groups.loc[dados_groups["group"] == i]
    dados_group_i[["genero", "vendas"]].groupby(["genero"]).sum().plot.bar()
    plt.show()

# Liberando variáveis temporárias
del dados_clean, dados_clean_category, dados_clean_norm, clusters, dados_clean_kmeans, dt1, new_dataframe, i, dados_group_i