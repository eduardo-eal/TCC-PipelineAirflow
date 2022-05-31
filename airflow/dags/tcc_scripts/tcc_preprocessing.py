import pandas as pd
from sklearn import preprocessing
import pandas as pd

pathScript = "~/airflow/dags/tcc_scripts"
pathScriptFeatureStore = "./featurestore"
pathTrue =  "~/airflow/dags/tcc_scripts/featurestore/True.csv"
pathFake = "~/airflow/dags/tcc_scripts/featurestore/Fake.csv"

noticias_verdadeiras = pd.read_csv(pathTrue);
noticias_falsas = pd.read_csv(pathFake);

#Merge das notícias
noticias_verdadeiras['fake']=0
noticias_falsas['fake']=1
noticias = pd.concat([noticias_verdadeiras,noticias_falsas])
noticias = noticias.sample(frac=1, random_state=120).reset_index(drop=True)

#Limpeza do texto
import texthero as hero

#Alterar todas as palavras para minúsculas
noticias['text'] = hero.lowercase(noticias['text'])

noticias['title'] = hero.lowercase(noticias['title'])

#Remover acentos
noticias['text'] = hero.remove_diacritics(noticias['text'])

noticias['title'] = hero.remove_diacritics(noticias['title'])

#Remover dígitos
noticias['text'] = hero.remove_digits(noticias['text'])

noticias['title'] = hero.remove_digits(noticias['title'])


#Remover chaves, parênteses, colchetes...
noticias['text'] = hero.remove_brackets(noticias['text'])
noticias['text'] = hero.remove_angle_brackets(noticias['text'])
noticias['text'] = hero.remove_curly_brackets(noticias['text'])
noticias['text'] = hero.remove_round_brackets(noticias['text'])
noticias['text'] = hero.remove_square_brackets(noticias['text'])

noticias['title'] = hero.remove_brackets(noticias['title'])
noticias['title'] = hero.remove_angle_brackets(noticias['title'])
noticias['title'] = hero.remove_curly_brackets(noticias['title'])
noticias['title'] = hero.remove_round_brackets(noticias['title'])
noticias['title'] = hero.remove_square_brackets(noticias['title'])

#remover stopwords
noticias['text'] = hero.remove_stopwords(noticias['text'])

noticias['title'] = hero.remove_stopwords(noticias['title'])

#remover URL
noticias['text'] = hero.remove_urls(noticias['text'])

noticias['title'] = hero.remove_urls(noticias['title'])

#Remover pontuação
noticias['text'] = hero.remove_punctuation(noticias['text'])

noticias['title'] = hero.remove_punctuation(noticias['title'])

#remover espaços em branco
noticias['text'] = hero.remove_whitespace(noticias['text'])

noticias['title'] = hero.remove_whitespace(noticias['title'])

noticias['text'] = noticias['text'].apply(lambda x: ' '.join([termo for termo in x.split() if len(termo) > 2 and termo != 'reuters']))
noticias['title'] = noticias['title'].apply(lambda x: ' '.join([termo for termo in x.split() if len(termo) > 2 and termo != 'reuters']))

#campo noticia_tratada vai conter o título seguido do texto
noticias['noticia_tratada'] = noticias['title']+' '+noticias['text']

noticias.drop(columns=['subject','date','text','title'], inplace=True)

#Lemmatizar
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner'])

#Retorna o texto com a lematização em formato de lista de tokens
def f_lemmatize(texto):
   return [termo.lemma_ for termo in nlp(texto)]

def token_para_texto(termos):
   return " ".join(termos)

noticias['noticia_tratada_tokens'] = noticias['noticia_tratada'].apply(f_lemmatize)
noticias['noticia_tratada_lemma'] = noticias['noticia_tratada_tokens'].apply(token_para_texto)
noticias.drop(columns=['noticia_tratada', 'noticia_tratada_tokens'], inplace=True)

#SEPARAR TREINO E TESTE
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(noticias['noticia_tratada_lemma'], 
                                              noticias['fake'], 
                                              stratify=noticias['fake'], 
                                              random_state=999)


#TF-IDF
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000) 
tfidf.fit(X_treino)
X_treino_tfidf = tfidf.transform(X_treino)
X_teste_tfidf = tfidf.transform(X_teste)
#noticias.drop(columns=['noticia_tratada_lemma'], inplace=True)
#return X_treino_tfidf, X_teste_tfidf, y_treino, y_teste

#SALVAR ARQUIVOS PARA PROXIMA ETAPA

import pickle

outfile1 = open(pathScriptFeatureStore+'/X_treino_tfidf.sm','wb')
pickle.dump(X_treino_tfidf,outfile1)
outfile1.close()

y_treino.to_csv(pathScriptFeatureStore+'/y_treino.csv', index=False)

outfile2 = open(pathScriptFeatureStore+'/X_teste_tfidf.sm','wb')
pickle.dump(X_teste_tfidf,outfile2)
outfile2.close()

y_teste.to_csv(pathScriptFeatureStore+'/y_teste.csv', index=False)
