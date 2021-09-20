#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:15:11 2021

@author: xinwenni
"""
##########################################################
# Content:
# part I:   Install and import packages and load data
# part II:  Pre-process and vectorize the documents
# Part III: Training LDA model
# Part IV:  Find the optimal number of topics using coherence_values
# Part V:   Compute similarity of topics 
# Part VI:  Visualize the topics

##########################################################

##########################################################

# Part I: install and import packages and load data  

##########################################################
# please install these modules before you run the code:
#!pip install gensim
#!pip install pandas
#!pip install nltk
#!pip install matplotlib



import os
import re
import pandas as pd
## `nltk.download('punkt')
import numpy as np

# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
import nltk
nltk.download('stopwords')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
#from gensim.test.utils import common_corpus
from gensim.models import LdaModel
from gensim.test.utils import datapath
#from gensim.models import LdaSeqModel
#from gensim.corpora import Dictionary, bleicorpus
#from gensim import models, similarities

# spacy for lemmatization
import spacy
from scipy.stats import wasserstein_distance

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import matplotlib.pyplot as plt
from collections import  defaultdict
from pprint import pprint
############################
# Please change the working directory to your path!
# os.chdir("/Users/xinwenni/LDA-DTM/DTM") 
############################

# load data
#df = pd.read_csv('df02.csv',encoding="ISO-8859-1")
df = pd.read_csv('dfsent02.csv',encoding="ISO-8859-1")

print(len(df))
print(df.iloc[0,:])

##########################################################

# Part II: Pre-process and vectorize the documents

##########################################################

# Convert to list
data= df.body.values.tolist()


def clean_data(data):
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    
    pprint(data[:1])
    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_lemm(data):
    data_words = list(sent_to_words(data))
#    print(data_words[:1])
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # See trigram example
#    print(trigram_mod[bigram_mod[data_words[0]]])
    
    # Remove Stop Words
    data_words_nostops=remove_stopwords(data_words)

    # Form Bigrams
#    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
#    print(data_lemmatized[:1])
    return data_lemmatized

# simple clean the data first 
data=clean_data(data)
# define stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'use','also'])
nlp = spacy.load('en', disable=['parser', 'ner'])

# Tokenize and lemmatize data 
data_lemmatized=get_lemm(data)

# Create Dictionary
id2word= corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

#Let’s see how many tokens and documents we have to train on.
print('Number of unique tokens: %d' % len(id2word))
print('Number of documents: %d' % len(corpus))

##########################################################

# Part III: Training LDA model 

##########################################################

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.
#
## Make a index to word dictionary.
#temp = dictionary[0]  # This is only to "load" the dictionary.
#id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
pprint(top_topics)


##########################################################

# Part IV: find the optimal number of topics using coherence_values

##########################################################
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
#        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=num_topics, id2word=id2word,random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=15,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=3, limit=17, step=2)

# Show graph
filename='Num_Topic_CV'
limit=17; start=3; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
#plt.ylim([0.25,0.45])
#plt.legend(("coherence_values"), loc='best')
plt.savefig(filename,dpi = 720,transparent=True)
plt.show()

# optimal number is 11, then adjust the topic number, and retrain the LDA model 
num_topic=11
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topic, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=15,
                                           alpha='auto',
                                           per_word_topics=True,
                                           minimum_probability=0.0)


lda_model.show_topics()

# Save model to disk.
temp_file = datapath("lda_model")
lda_model.save(temp_file)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics(0,10))
doc_lda = lda_model[corpus]
 
##########################################################

# Part V: Compute similarity of topics 

##########################################################

def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)


try:
    get_ipython()
    import plotly.offline as py
except Exception:
    #
    # Fall back to matplotlib if we're not in a notebook, or if plotly is
    # unavailable for whatever reason.
    #
    plot_difference = plot_difference_matplotlib
else:
    py.init_notebook_mode()
    plot_difference = plot_difference_plotly
    
mdiff, annotation = lda_model.diff(lda_model, distance='hellinger', num_words=50)
plot_difference(mdiff,  annotation=annotation)
plt.tick_params(labelsize=23)
plt.savefig("topic_distance_H.png",dpi = 360,transparent=True)


##########################################################

# Part VI: Visualize the topics

##########################################################

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.show(vis)
##########################################################
