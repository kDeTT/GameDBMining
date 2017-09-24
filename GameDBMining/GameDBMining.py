import pandas as pd
import math

# Carregando os dados
dados = pd.read_csv("dados-07.csv")

# Substituindo valores 'tbd' por 'nan' no atributo 'avaliacao-usuarios'
dados['avaliacao-usuarios']= dados['avaliacao-usuarios'].replace({'tbd':'-999'})
dados['avaliacao-usuarios'] = dados['avaliacao-usuarios'].astype(float)
dados['avaliacao-usuarios'] = dados['avaliacao-usuarios'].replace({-999:math.nan})

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
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if (math.isnan(game[3]) and ((game[0], game[1]) in dados_filter2_mean)):
        game[3] = dados_filter2_mean[{game[0], game[1]}].values[0]
        dados_filter1.set_value(i, 'avaliacao-criticos', game[3])

# Faz a substituição dos valores ausentes de avaliação dos usuários pela média de avaliação por editora e gênero
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if (math.isnan(game[5]) and ((game[0], game[1]) in dados_filter3_mean)):
        game[5] = dados_filter3_mean[{game[0], game[1]}].values[0]
        dados_filter1.set_value(i, 'avaliacao-usuarios', game[5])
        
# Filtrando os dados_filter_1 por editora, gênero e quantidade de críticos
dados_filter2 = dados_filter1_clean[["editora", "genero", "numero-criticos"]]

# Média de avaliação dos críticos por editora e gênero
dados_filter2_mean = dados_filter2.groupby(["editora", "genero"])["numero-criticos"].mean()

# Filtrando os dados_filter_1 por editora, gênero e quantidade de usuários
dados_filter3 = dados_filter1_clean[["editora", "genero", "numero-usuarios"]]

# Média de avaliação dos usuários por editora e gênero
dados_filter3_mean = dados_filter3.groupby(["editora", "genero"])["numero-usuarios"].mean()
        
# Faz a substituição dos valores ausentes de avaliação dos críticos pela média de avaliação por editora e gênero
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if (math.isnan(game[2]) and ((game[0], game[1]) in dados_filter2_mean)):
        game[2] = dados_filter2_mean[{game[0], game[1]}].values[0]
        dados_filter1.set_value(i, "numero-criticos", game[2])

# Faz a substituição dos valores ausentes de avaliação dos usuários pela média de avaliação por editora e gênero
for i in range(0, len(dados_filter1.values)):
    game = dados_filter1.values[i]
    
    if (math.isnan(game[4]) and ((game[0], game[1]) in dados_filter3_mean)):
        game[4] = dados_filter3_mean[{game[0], game[1]}].values[0]
        dados_filter1.set_value(i, "numero-usuarios", game[4])

# Ajusta os dados originais
dados["numero-criticos"] = dados_filter1["numero-criticos"]
dados["avaliacao-criticos"] = dados_filter1["avaliacao-criticos"]
dados["numero-usuarios"] = dados_filter1["numero-usuarios"]
dados["avaliacao-usuarios"] = dados_filter1["avaliacao-usuarios"]

dados_clean = dados.dropna(subset=["editora", "genero", "numero-criticos", "avaliacao-criticos", "numero-usuarios", "avaliacao-usuarios"])

del dados_filter1, dados_filter1_clean, dados_filter2, dados_filter2_mean, dados_filter3, dados_filter3_mean, game, i