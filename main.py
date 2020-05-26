#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[347]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import (KBinsDiscretizer,OneHotEncoder,StandardScaler)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer,TfidfVectorizer,TfidfTransformer)


# In[98]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("countries.csv",decimal=',')


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[6]:


#remove blank space 
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()
countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.
#q1()
countries['Region'].sort_values().unique()


# In[89]:


#q2() - Discretizando a variável Pop_density em 10 intervalos com KBinsDiscretizer, seguindo o encode ordinal e estratégia quantile,#quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.
def get_inverval(bin_idx,bin_edges):
    return f"{np.round(bin_edges[bin_idx],2)} ⊢ {np.round(bin_edges[bin_idx+1],2)}"

discretizer = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
discretizer.fit(countries[['Pop_density']])
score_bins = discretizer.transform(countries[['Pop_density']])

print("\nSolução Kazuki")
for i in range(len(discretizer.bin_edges_[0])-1):
    print(f"{get_inverval(i,discretizer.bin_edges_[0])}: {sum(score_bins[:,0] == i)}")

print("\nResposta utilizando Dataframe")
dis_coutries = countries.copy()
dis_coutries['bins'] = score_bins
int(dis_coutries.groupby('bins').count().loc[9]['Pop_density'])


# In[126]:


#Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.
q3_df = countries.copy()
imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
inputed_climate = imputer.fit_transform(countries[['Climate']])
q3_df['Climate'] = inputed_climate
one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
one_hot_encoder.fit_transform(q3_df[['Region']])
att_region = len(one_hot_encoder.categories_[0])
one_hot_encoder.fit_transform(q3_df[['Climate']])
att_climate = len(one_hot_encoder.categories_[0])
att_climate + att_region


# In[329]:


# Questão 4
#Aplique o seguinte _pipeline_:
#1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
#2. Padronize essas variáveis.
#Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.
test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]
#create test DF
df_test = pd.DataFrame([test_country],columns=countries.columns)

#select numeric columns
num_columns = countries.select_dtypes(include=['int64','float64']).columns

#create numeric pipeline
num_pipe = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('standard',StandardScaler())
])

#create a preprocessor in numeric columns
preprocessor = ColumnTransformer(transformers=[
    ('numeric_feature',num_pipe,num_columns)
])

#apply preprocessor in countries df
fit_pre = preprocessor.fit(countries)

#apply preprocessor in test df
test_result = fit_pre.transform(df_test)

round(test_result[0][9],3)


# In[345]:


#q5 - Remoção de outliers da coluna Net_migration
net_migration = countries.Net_migration.copy()
qt1 = net_migration.quantile(0.25)
qt3 = net_migration.quantile(0.75)
iqr = qt3 - qt1
non_outlier = [qt1-1.5*iqr,qt3+1.5*iqr]
outliers = (len(net_migration[net_migration<non_outlier[0]]),
              len(net_migration[net_migration>non_outlier[1]]), False)
outliers


# In[348]:


#q6 - contar palavra phone
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[364]:


#q6
count_vectorizer = CountVectorizer()
newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
phone_idx = count_vectorizer.vocabulary_.get('phone')
pd.DataFrame(newsgroup_counts[:,phone_idx].toarray(),columns=['phone']).sum()[0]


# In[372]:


#q7
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(newsgroup.data)
ng_tfidf_vec = tfidf_vectorizer.transform(newsgroup.data)
np.round(pd.DataFrame(ng_tfidf_vec[:,phone_idx].toarray(),columns=['phone']).sum()[0],3).item()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[99]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return list(countries['Region'].sort_values().unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[85]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
    discretizer.fit(countries[['Pop_density']])
    score_bins = discretizer.transform(countries[['Pop_density']])
    return int(dis_coutries.groupby('bins').count().loc[9]['Pop_density'])


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[129]:


def q3():
    q3_df = countries.copy()
    imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
    inputed_climate = imputer.fit_transform(countries[['Climate']])
    q3_df['Climate'] = inputed_climate
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
    one_hot_encoder.fit_transform(q3_df[['Region']])
    att_region = len(one_hot_encoder.categories_[0])
    one_hot_encoder.fit_transform(q3_df[['Climate']])
    att_climate = len(one_hot_encoder.categories_[0])
    return (att_climate + att_region)


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[332]:


def q4():
    #create test DF
    df_test = pd.DataFrame([test_country],columns=countries.columns)

    #select numeric columns
    num_columns = countries.select_dtypes(include=['int64','float64']).columns

    #create numeric pipeline
    num_pipe = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('standard',StandardScaler())
    ])

    #create a preprocessor in numeric columns
    preprocessor = ColumnTransformer(transformers=[
        ('numeric_feature',num_pipe,num_columns)
    ])

    #fit preprocessor in train df
    fit_pre = preprocessor.fit(countries)

    #apply preprocessor in test df
    test_result = fit_pre.transform(df_test)
    return round(test_result[0][9],3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[346]:


def q5():
    net_migration = countries.Net_migration.copy()
    qt1 = net_migration.quantile(0.25)
    qt3 = net_migration.quantile(0.75)
    iqr = qt3 - qt1
    non_outlier = [qt1-1.5*iqr,qt3+1.5*iqr]
    outliers = (len(net_migration[net_migration<non_outlier[0]]),
                len(net_migration[net_migration>non_outlier[1]]), False)
    return outliers


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[13]:


def q6():
    count_vectorizer = CountVectorizer()
    newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
    phone_idx = count_vectorizer.vocabulary_.get('phone')
    phone = pd.DataFrame(newsgroup_counts[:,phone_idx].toarray(),columns=['phone']).sum()[0]
    return phone


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[375]:


def q7():
    count_vectorizer = CountVectorizer()
    newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
    phone_idx = count_vectorizer.vocabulary_.get('phone')
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroup.data)
    ng_tfidf_vec = tfidf_vectorizer.transform(newsgroup.data)
    tfidf_phone = np.round(pd.DataFrame(ng_tfidf_vec[:,phone_idx].toarray(),columns=['phone']).sum()[0],3).item() 
    return tfidf_phone
q7()

