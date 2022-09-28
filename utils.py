import pandas as pd
import pandas_gbq
import numpy as np
import glob, os
import seaborn as sns
import matplotlib.cm
import matplotlib.colors
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from keras.models import load_model
import json
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from keras.utils import plot_model
from keras import regularizers

from gensim.models.keyedvectors import KeyedVectors

from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from keras.models import Sequential
from keras.layers import concatenate, \
Activation, Dense, Dropout, BatchNormalization, Embedding, Input, LSTM,Conv1D, Concatenate, Reshape,Conv2D, MaxPool2D, MaxPooling1D,Flatten, Embedding
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
# from nltk.corpus import stopwords
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from IPython.display import HTML


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF for better performance")


def plot_history(history, metric_train, metric_val):
    acc = history.history[metric_train]
    val_acc = history.history[metric_val]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training {}'.format(metric_train))
    plt.plot(x, val_acc, 'r', label='Validation {}'.format(metric_val))
    plt.title('Training and validation {}'.format(metric_train))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def generate_embeddings(df_m):
  inst_audio = []
  inst_video = []

  for emb_list in df_m['audio_embeddings'].values:
    inst_audio.append(emb_list)

  for emb_list in df_m['video_embeddings'].values:
    inst_video.append(emb_list)

  X_audio = np.array(inst_audio)
  X_video = np.array(inst_video)
  # print(X_audio[0:2])
  # print(X_video[0:2])
  #X_video = X_video.reshape(X_video.shape[0],len(inst_video[0]))
  print(X_audio.shape)
  print(X_video.shape)
  X_av = np.hstack((X_audio, X_video))


  print(X_av.shape)

  return (X_audio, X_video, X_av)

def generate_scaled(X):
  X_scaled = MinMaxScaler().fit_transform(X)

  return X_scaled

def generated_tsne_3d(X, y):
  tsne_obj = TSNE(n_components=3, random_state=0).fit_transform(X)

  tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'Z':tsne_obj[:,2],
                        'title':y})
  return tsne_df

def aggregate_title_embeddings(df, X):
  list_feats = []

  title_list = list(df['title_id'].unique())

  for tit in title_list:
    videos_indexes = list(df[df['title_id']==tit].index)
    
    emb_title = np.mean(X[videos_indexes], axis=0)

    list_feats.append(emb_title)

  return (np.array(list_feats), title_list)

def generate_recommendations_nn(model_trained, df, X, top_n=10):
  # Inferência de embeddings dos vídeos usando rede neural densa
  X_emb_predicted  = model_trained.predict(X)

  # Inferência de embeddings dos vídeos usando PCA
  #X_emb_pred_pca = generate_pca_feature(X)

  # Aplicando um MinMaxScaler a fim de obter uma nova escala de valores entre 0 e 1
  X_emb_pred_scaled = generate_scaled(X_emb_predicted)

  # Obtendo os embeddings agregados por tíítulo
  X_emb_title_nn, y_title_nn = aggregate_title_embeddings(df, X_emb_pred_scaled)
  #X_emb_title_simple, y_title_simple = aggregate_title_embeddings(df, X)
  #X_emb_title_pca, y_title_pca = aggregate_title_embeddings(df, X_emb_pred_pca)

  # Obtendo as recomendações para cada tipo de embedding
  rec_nn     = get_recommnedations(X_emb_title_nn, np.array(y_title_nn))
  #rec_nn     = get_recommnedations_cosine(X_emb_title_nn, np.array(y_title_nn))

  return rec_nn

def get_recommnedations(X, y, n_neighs=100):

  neigh = KNeighborsClassifier(n_neighbors=n_neighs, metric='cosine')
  neigh.fit(X, y)

  distances, neigh_ids = neigh.kneighbors(X, n_neighs)

  titles_recommendations = {}

  for i, title in enumerate(y):
    titles_scores = [distances[i]]
    titles_neighs = [neigh_ids[i]]
    normalized_scores = 1 - normalize(titles_scores)
    # #normalized_scores = normalize(titles_scores)

    grouped_title_scores = {}
    for idx, titles_id in enumerate(titles_neighs):
      neighs_title_name = y[tuple(titles_neighs)]
      neighs_title_distance = normalized_scores[idx]

      for neigh_title, neigh_distance in zip(neighs_title_name, neighs_title_distance):
        grouped_title_scores.setdefault(neigh_title, []).append(neigh_distance)

    titles_recommendations[title] = sorted(
      [(tit, dist[0]) for tit, dist in grouped_title_scores.items()],
      key=lambda x: x[1],
      reverse=True
    )
  return titles_recommendations

def get_recommnedations_cosine(X, y, n_neighs=100):

  X_similarities = cosine_similarity(X)

  titles_recommendations = {}

  for i, title_query in enumerate(y):
    neighbors = X_similarities[i]
    grouped_title_scores = {}
    for j, neigh_title in enumerate(y):
      grouped_title_scores.setdefault(neigh_title, []).append(neighbors[j])
    
    titles_recommendations[title_query] = sorted(
      [(tit, dist[0]) for tit, dist in grouped_title_scores.items()],
      key=lambda x: x[1],
      reverse=True
    )[1:n_neighs]

  return titles_recommendations

# Extração de features com PCA
def generate_pca_feature(X):
  pca = PCA(n_components=500)
  return pca.fit(X).transform(X)

def load_from_list(list_values):
  result = []
  for v in list_values:
    result.append(v['element'])
  return result

def fetch_preview_table_df(title_recs, df_inf):

  url_template = "https://globoplay.globo.com/v/t/{}/"

  row_element_template = """
  <tr><td><img src="{}" width="100" height="150"></td>
  <td><h4><a href="{}">{}</a></h4>
     <p>{}</p>
     <p><b>Tipo:</b>{}</p>
     <p><b>Classificação:</b>{} - {}</p>
     <p><b>Genero:</b>{}</p>
     <p><b>Score:</b>{}</p>
  </td></tr>
  """
  rows = []

  table_str = """
  <table>{}</table>
  """

  for title_id, score in title_recs:
    title_df = df_inf[df_inf['title_id']==title_id]
    title_name = title_df['title_name'].values[0]
    title_preview_name = title_df['title_preview_name'].values[0]
    description = title_df['title_description'].values[0]
    image = title_df['title_cover'].values[0]

    metadata_type = title_df['metadata_type'].values[0]
    content_rating = title_df['content_rating'].values[0]

    content_rating_criteria = title_df['content_rating_criteria'].values[0]
    genres_names = title_df['genres_names'].values[0]

    content_rating_criteria_list = load_from_list(content_rating_criteria)
    genres_names_list = load_from_list(genres_names)


    #url_query = df_inf[df_inf['title_name']==title_name]['url_globoplay'].values[0]
    url_query = url_template.format(title_df['title_id'].values[0])

    if title_preview_name == 'Ops...':
      continue
    row_element = row_element_template.format(image, url_query, title_name, description, 
                                              metadata_type, content_rating, content_rating_criteria_list, genres_names_list, score)
    rows.append(row_element)

  return table_str.format("".join(rows))

def print_recommendations(model_trained, df, X, query, top_n=10):

  # Inferência de embeddings dos vídeos usando rede neural densa
  X_emb_predicted  = model_trained.predict(X)

  # Inferência de embeddings dos vídeos usando PCA
  X_emb_pred_pca = generate_pca_feature(X)

  # Aplicando um MinMaxScaler a fim de obter uma nova escala de valores entre 0 e 1
  X_emb_pred_scaled = generate_scaled(X_emb_predicted)

  # Obtendo os embeddings agregados por tíítulo
  X_emb_title_nn, y_title_nn = aggregate_title_embeddings(df, X_emb_pred_scaled)
  X_emb_title_simple, y_title_simple = aggregate_title_embeddings(df, X)
  X_emb_title_pca, y_title_pca = aggregate_title_embeddings(df, X_emb_pred_pca)

  # Obtendo as recomendações para cada tipo de embedding
  rec_simple = get_recommnedations(X_emb_title_simple, np.array(y_title_simple))
  rec_nn     = get_recommnedations(X_emb_title_nn, np.array(y_title_nn))
  rec_pca    = get_recommnedations(X_emb_title_pca, np.array(y_title_pca))

  print('Titulos similares a: {}'.format(query))

  return pd.DataFrame({'ranking_emb_simple({})'.format(X_emb_title_simple.shape[1]):list(rec_simple[query][1:top_n]), 
                       'ranking_emb_nn({})'.format(X_emb_title_nn.shape[1]):list(rec_nn[query][1:top_n]),
                       'ranking_emb_pca({})'.format(X_emb_title_pca.shape[1]):list(rec_pca[query][1:top_n])})

def print_recommendations_html(recommendations_mod, df_tf, query, top_n=10):


  title_df = df_tf[df_tf['title_id']==query]

  url = "https://globoplay.globo.com/v/t/{}/".format(title_df['title_id'].values[0])
  description = title_df['title_description'].values[0]
  image = title_df['title_cover'].values[0]
  metadata_type = title_df['metadata_type'].values[0]

  content_rating = title_df['content_rating'].values[0]

  content_rating_criteria = title_df['content_rating_criteria'].values[0]
  genres_names = title_df['genres_names'].values[0]

  content_rating_criteria_list = load_from_list(content_rating_criteria)
  genres_names_list = load_from_list(genres_names)

  html_query = """<h3>Título Base:</h3><table><tr><td><img src="{}" width="100" height="150">
  </td><td><h4><a href="{}">{}</a></h4>
     <p>{}</p>
     <p><b>Tipo:</b>{}</p>
     <p><b>Classificação:</b>{} - {}</p>
     <p><b>Genero:</b>{}</p>
  </td></tr></table>
  <tr><td><h3>TOP {} Titulos Similares:</h3></td></tr>""".format(image,url,query, description, 
                                              metadata_type, content_rating, content_rating_criteria_list, genres_names_list, top_n)


  html_recs = fetch_preview_table_df(list(recommendations_mod[query][1:top_n]), df_tf)


  html_output = """<html><body>{v1}<div></div>{v2}</body></html>
    """.format(v1=html_query,v2=html_recs)

  return html_output

def fetch_preview_json_df(title_recs, df_inf):

  url_template = "https://globoplay.globo.com/v/t/{}/"
  rows = []
  table_str = """
  <table>{}</table>
  """

  for title_id, score in title_recs:

    df_filtered_query = df_inf[df_inf['title_id']==title_id]

    title_preview_name = df_filtered_query['title_preview_name'].values[0]
    title_name           = df_filtered_query['title_name'].values[0]
    description        = df_filtered_query['title_description'].values[0]
    image              = df_filtered_query['title_cover'].values[0]

    metadata_type  = df_filtered_query['metadata_type'].values[0]
    content_rating = df_filtered_query['content_rating'].values[0]

    content_rating_criteria = df_filtered_query['content_rating_criteria'].values[0]
    genres_names            = df_filtered_query['genres_names'].values[0]
    countries_names         = df_filtered_query['contries'].values[0]
    director_names          = df_filtered_query['director_names'].values[0]
    release_year            = df_filtered_query['release_year'].values[0]
    cast_names              = df_filtered_query['cast_names'].values[0]


    content_rating_criteria_list = load_from_list(content_rating_criteria)
    genres_names_list            = load_from_list(genres_names)
    countries_names_list         = load_from_list(countries_names)
    director_names_list          = load_from_list(director_names)
    cast_names_list              = load_from_list(cast_names)
    release_year                 = int(release_year) if not np.isnan(release_year) else None
    description                  = description if not type(description)==float else None
  

    #url_query = df_inf[df_inf['title_name']==title_name]['url_globoplay'].values[0]
    url_query = url_template.format(df_filtered_query['title_id'].values[0])

    if title_preview_name == 'Ops...':
      continue


    row_element = {'title_id':title_id, 
                   'title_name':title_name,
                   'url':url_query,
                  'description':description,
                 'image':image,
                 'metadata_type':metadata_type,
                 'content_rating':content_rating,
                 'content_rating_criteria':content_rating_criteria_list,
                 'genre':genres_names_list,
                 'country':countries_names_list,
                 'director':director_names_list,
                 'cast':cast_names_list,
                 'release_year':release_year,
                 'score':score}

    # row_element = row_element_template.format(image, url_query, title_name, description, 
    #                                           metadata_type, content_rating, content_rating_criteria_list, genres_names_list)
    rows.append(row_element)

  #return table_str.format("".join(rows))
  return rows

def fetch_preview_json_df_simple(title_recs, df_inf):

  url_template = "https://globoplay.globo.com/v/t/{}/"
  rows = []

  table_str = """
  <table>{}</table>
  """

  for title_id, score in title_recs:
    title_df = df_inf[df_inf['title_name']==title_id]
    title_name = title_df['title_name'].values[0]
    title_preview_name = title_df['title_preview_name'].values[0]
    description = title_df['title_description'].values[0]
    image = title_df['title_cover'].values[0]

    metadata_type = title_df['metadata_type'].values[0]
    content_rating = title_df['content_rating'].values[0]

    content_rating_criteria = title_df['content_rating_criteria'].values[0]
    genres_names = title_df['genres_names'].values[0]

    content_rating_criteria_list = load_from_list(content_rating_criteria)
    genres_names_list = load_from_list(genres_names)
    #score = df_inf[df_inf['title_name']==title_name]['score'].values[0]


    #url_query = df_inf[df_inf['title_name']==title_name]['url_globoplay'].values[0]
    url_query = url_template.format(title_df['title_id'].values[0])

    if title_preview_name == 'Ops...':
      continue


    row_element = {'title_id':title_id, 'score':score}

    # row_element = row_element_template.format(image, url_query, title_name, description, 
    #                                           metadata_type, content_rating, content_rating_criteria_list, genres_names_list)
    rows.append(row_element)

  #return table_str.format("".join(rows))
  return rows

def print_recommendations_json(recommendations_mod, df_tf, query, top_n=10):

  url = "https://globoplay.globo.com/v/t/{}/".format(df_tf[df_tf['title_name']==query]['title_id'].values[0])
  description = df_tf[df_tf['title_name']==query]['title_description'].values[0]
  image = df_tf[df_tf['title_name']==query]['title_cover'].values[0]
  metadata_type = df_tf[df_tf['title_name']==query]['metadata_type'].values[0]

  content_rating = df_tf[df_tf['title_name']==query]['content_rating'].values[0]

  content_rating_criteria = df_tf[df_tf['title_name']==query]['content_rating_criteria'].values[0]
  genres_names = df_tf[df_tf['title_name']==query]['genres_names'].values[0]

  content_rating_criteria_list = load_from_list(content_rating_criteria)
  genres_names_list = load_from_list(genres_names)

  # html_query = """<h3>Título Base:</h3><table><tr><td><img src="{}" width="100" height="150">
  # </td><td><h4><a href="{}">{}</a></h4>
  #    <p>{}</p>
  #    <p><b>Tipo:</b>{}</p>
  #    <p><b>Classificação:</b>{} - {}</p>
  #    <p><b>Genero:</b>{}</p>
  # </td></tr></table>
  # <tr><td><h3>TOP {} Titulos Similares:</h3></td></tr>""".format(image,url,query, description, 
  #                                             metadata_type, content_rating, content_rating_criteria_list, genres_names_list, top_n)
  
  titulo_base = {'title_name':query,
                 'url':url,
                 'description':description,
                 'image':image,
                 'metadata_type':metadata_type,
                 'content_rating':content_rating,
                 'content_rating_criteria':content_rating_criteria_list,
                 'genre':genres_names_list}
  

  # html_recs = fetch_preview_json_df(list(recommendations_mod[query][1:top_n]), df_tf)
  recs = fetch_preview_json_df(list(recommendations_mod[query][:top_n]), df_tf)

  # html_output = """<html><body>{v1}<div></div>{v2}</body></html>
  #   """.format(v1=html_query,v2=html_recs)

  json_output = {'base_title':titulo_base, 'recommendations':recs}

  return json_output

def print_recommendations_json_simple(recommendations_mod, df_tf, query, top_n=10):

  df_filtered_query = df_tf[df_tf['title_id']==query]

  url = "https://globoplay.globo.com/v/t/{}/".format(df_filtered_query['title_id'].values[0])

  description        = df_filtered_query['title_description'].values[0]
  title_name           = df_filtered_query['title_name'].values[0]
  title_preview_name = df_filtered_query['title_preview_name'].values[0]
  image              = df_filtered_query['title_cover'].values[0]
  metadata_type      = df_filtered_query['metadata_type'].values[0]

  content_rating     = df_filtered_query['content_rating'].values[0]

  content_rating_criteria = df_filtered_query['content_rating_criteria'].values[0]
  genres_names            = df_filtered_query['genres_names'].values[0]
  countries_names         = df_filtered_query['contries'].values[0]
  director_names          = df_filtered_query['director_names'].values[0]
  release_year            = df_filtered_query['release_year'].values[0]
  cast_names              = df_filtered_query['cast_names'].values[0]


  content_rating_criteria_list = load_from_list(content_rating_criteria)
  genres_names_list            = load_from_list(genres_names)
  countries_names_list         = load_from_list(countries_names)
  director_names_list          = load_from_list(director_names)
  cast_names_list              = load_from_list(cast_names)
  release_year                 = int(release_year) if not np.isnan(release_year) else None
  description                  = description if not type(description)==float else None
  
  if title_preview_name == 'Ops...':
    return None

  titulo_base = {'title_id':query,
                 'title_name':title_name,
                 'url':url,
                 'description':description,
                 'image':image,
                 'metadata_type':metadata_type,
                 'content_rating':content_rating,
                 'content_rating_criteria':content_rating_criteria_list,
                 'genre':genres_names_list,
                 'country':countries_names_list,
                 'director':director_names_list,
                 'cast':cast_names_list,
                 'release_year':release_year}
  
  recs = fetch_preview_json_df(list(recommendations_mod[query][:top_n]), df_tf)

  json_output = {'base_title':titulo_base, 'recommendations':recs}

  return json_output

def print_title_json_simple(df_tf, query):

  title_df = df_tf[df_tf['title_id']==query]
  url = "https://globoplay.globo.com/v/t/{}/".format(title_df['title_id'].values[0])
  description = title_df['title_description'].values[0]
  title_name = title_df['title_name'].values[0]
  title_preview_name = title_df['title_preview_name'].values[0]
  image = title_df['title_cover'].values[0]
  metadata_type = title_df['metadata_type'].values[0]

  content_rating = title_df['content_rating'].values[0]

  content_rating_criteria = title_df['content_rating_criteria'].values[0]
  genres_names = title_df['genres_names'].values[0]

  content_rating_criteria_list = load_from_list(content_rating_criteria)
  genres_names_list = load_from_list(genres_names)

  if title_preview_name == 'Ops...':
    return None

  titulo_base = {'title_id':query,
                 'title_name':title_name,
                 'image':image}

  return titulo_base


def generate_multiclass(row):
  genres_names = row['genres_names']
  content_rating = row['content_rating']
  content_rating_criteria = row['content_rating_criteria']

  genres_list         = load_from_list(genres_names) if len(genres_names) > 0 else ['unknow_genre']
  content_rating_list = [content_rating] if content_rating is not None else ['99']
  content_rating_criteria_list = load_from_list(content_rating_criteria) if len(content_rating_criteria)> 0 else ['unknow_criteria']

  return genres_list + content_rating_criteria_list + content_rating_list

def generate_multiclass_genre_only(row):
  genres_names = row['genres_names']
  #content_rating = row['content_rating']
  #content_rating_criteria = row['content_rating_criteria']

  genres_list         = load_from_list(genres_names) if len(genres_names) > 0 else ['unknow_genre']
  #content_rating_list = [content_rating] if content_rating is not None else ['99']
  #content_rating_criteria_list = load_from_list(content_rating_criteria) if len(content_rating_criteria)> 0 else ['unknow_criteria']

  return genres_list

# Criação do modelo de rede neural para a classificação de títulos
def make_model(n_features, n_classes):
  model = Sequential()
  model.add(Dense(1024, input_shape=(n_features,),
            kernel_initializer='glorot_normal',name='fc1'))
  model.add(BatchNormalization(name='bn1'))
  model.add(Activation('relu',name='ac1'))
  model.add(Dropout(0.5))
  model.add(Dense(512, kernel_initializer='glorot_normal', use_bias=False, name='fc2'))
  model.add(BatchNormalization(name='bn2'))
  model.add(Activation('relu',name='ac2'))
  model.add(Dropout(0.25))
  model.add(Dense(512, kernel_initializer='glorot_normal', use_bias=False, name='fc3'))
  model.add(BatchNormalization(name='bn3'))
  model.add(Activation('relu',name='ac3'))
  model.add(Dropout(0.15))
  model.add(Dense(256, kernel_initializer='glorot_normal', use_bias=False, name='fc4'))
  model.add(BatchNormalization(name='bn4'))
  model.add(Activation('relu',name='ac4'))
  model.add(Dropout(0.1))
  model.add(Dense(n_classes, activation='softmax',name='fc5'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


def make_model_multi(n_features, n_classes):
  model = Sequential()
  model.add(Dense(1024, input_shape=(n_features,),
            kernel_initializer='glorot_normal',name='fc1'))
  model.add(BatchNormalization(name='bn1'))
  model.add(Activation('relu',name='ac1'))
  model.add(Dropout(0.5))
  model.add(Dense(512, kernel_initializer='glorot_normal', use_bias=False, name='fc2'))
  model.add(BatchNormalization(name='bn2'))
  model.add(Activation('relu',name='ac2'))
  model.add(Dropout(0.25))
  model.add(Dense(512, kernel_initializer='glorot_normal', use_bias=False, name='fc3'))
  model.add(BatchNormalization(name='bn3'))
  model.add(Activation('relu',name='ac3'))
  model.add(Dropout(0.15))
  model.add(Dense(256, kernel_initializer='glorot_normal', use_bias=False, name='fc4'))
  model.add(BatchNormalization(name='bn4'))
  model.add(Activation('relu',name='ac4'))
  model.add(Dropout(0.1))

  model.add(Dense(n_classes, activation='sigmoid',name='fc5'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])
  return model

def make_model_multi_reg(n_features, n_classes):
  model = Sequential()
  model.add(Dense(1024, input_shape=(n_features,),
            kernel_initializer='glorot_normal', name='fc1'))
  model.add(BatchNormalization(name='bn1'))
  model.add(Activation('relu',name='ac1'))
  model.add(Dropout(0.5))
  model.add(Dense(512, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), use_bias=True, name='fc2'))
  model.add(BatchNormalization(name='bn2'))
  model.add(Activation('relu',name='ac2'))
  model.add(Dropout(0.25))
  model.add(Dense(512, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), use_bias=True, name='fc3'))
  model.add(BatchNormalization(name='bn3'))
  model.add(Activation('relu',name='ac3'))
  model.add(Dropout(0.15))
  model.add(Dense(256, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), use_bias=True, name='fc4'))
  model.add(BatchNormalization(name='bn4'))
  model.add(Activation('relu',name='ac4'))
  model.add(Dropout(0.1))

  model.add(Dense(n_classes, activation='sigmoid',name='fc5'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])
  return model


# Faz o balanceamento do dataset e o treino do modelo
def train_model(X, y, n_batch_size=128, n_epochs=40, n_validation_split=0.1):

  # Balanceando o dataset
  sm = SMOTE(random_state=42)
  ros = RandomOverSampler(random_state=42)

  X_res, y_res = ros.fit_resample(X, y)
  y_res = y_res.reshape(y_res.shape[0],1)

  encoder = OneHotEncoder()
  encoder.fit(y_res)
  encoded_y_res = encoder.transform(y_res)

  model = make_model(X_res.shape[1],encoded_y_res.shape[1])
  print(model.summary())

  history = model.fit(X_res, encoded_y_res, batch_size=n_batch_size, epochs=n_epochs, validation_split=n_validation_split)

  return (history, model)

def train_model_multilabel(X, y, n_batch_size=128, n_epochs=40, n_validation_split=0.1):

  # Balanceando o dataset
  # sm = SMOTE(random_state=42)
  # ros = RandomOverSampler(random_state=42)

  # X_res, y_res = ros.fit_resample(X, y)
  # y_res = y_res.reshape(y_res.shape[0],1)

  # encoder = OneHotEncoder()
  # encoder.fit(y_res)
  # encoded_y_res = encoder.transform(y_res)
  mlb = MultiLabelBinarizer()
  y_encoded = mlb.fit_transform(y)

  # ros = RandomOverSampler(random_state=42)

  # X_res, y_res = ros.fit_resample(X, y_encoded)
  print(y_encoded.shape)
  #y_res = y_res.reshape(y_res.shape[0],1)

  n_class = len(mlb.classes_)

  model = make_model_multi(X.shape[1], n_class)
  print(model.summary())

  history = model.fit(X, y_encoded, batch_size=n_batch_size, epochs=n_epochs, validation_split=n_validation_split)

  return (history, model)

def train_model_multilabel_reg(X, y, n_batch_size=128, n_epochs=40, n_validation_split=0.1):

  mlb = MultiLabelBinarizer()
  y_encoded = mlb.fit_transform(y)

  print(y_encoded.shape)

  n_class = len(mlb.classes_)

  model = make_model_multi_reg(X.shape[1], n_class)
  print(model.summary())

  history = model.fit(X, y_encoded, batch_size=n_batch_size, epochs=n_epochs, validation_split=n_validation_split)

  return (history, model)

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1
