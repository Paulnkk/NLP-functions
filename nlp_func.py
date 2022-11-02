import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
import seaborn as sns

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.util import ngrams
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from wordcloud import WordCloud

def split_city(df):    
    # filter by city
    nyc_df = df[df['SYSTEM_IDENTIFIER'] == 'newyork'] 
    print('NYC data rows:', len(nyc_df))

    sf_df = df[df['SYSTEM_IDENTIFIER'] == 'sanfrancisco']
    print('SF data rows:', len(sf_df))

    wash_df = df[df['SYSTEM_IDENTIFIER'] == 'washingtondc']
    print('WASH data rows:', len(wash_df))

    miami_df = df[df['SYSTEM_IDENTIFIER'] == 'miami']
    print('MIAMI data rows:', len(miami_df))

    # remove rows without timestamp
    nyc_df = nyc_df[nyc_df['SYSTEM_STARTED_AT'].notna()]
    
    return nyc_df, miami_df, wash_df, sf_df

# remove url function
def remove_url(text): 
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

# Remove Emojis
def remove_emoji(text): 
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# df - dataframe
# col - column name of review df as String
# col1 - column of SYSTEM_STARTED_AT
# col2 - column of NEXT_RIDE
# text cleaning function, removing URLs, emojis, lemmatization, remove punctuations, etc.
def clean_review_text(df, col, col1, col2):
    # Transform text to pure lower case text
    df[col] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove punctuations 
    df[col] = [re.sub('[^a-zA-Z]', ' ', str(i)) for i in df[col]]
    df[col] = [i.lower() for i in df[col]]

    # Remove url
    df[col] = df[col].apply(lambda x: remove_url(x))

    # Remove Emojis
    df[col] = df[col].apply(lambda x: remove_emoji(x))

    # split into words
    df[col] = [word_tokenize(i) for i in df[col]]

    # remove all tokens that are not alphabetic
    df[col] = [[i for i in j if i.isalpha()==True] for j in df[col]]

    # filter out stop words 
    
    # add stop words list -> angepasst, um wesentlichen Probleme auszuschließen
    add_words = ["revel", "please", "would", "could"]

    stop_words = list(stopwords.words('english'))

    for i in add_words:
        stop_words.append(i)
    
    stop_words = [i for i in stop_words if i != 'not']
    df[col] = [[i for i in j if not i in stop_words] for j in df[col]]

    # convert to lower case 
    df[col] = [[i.lower() for i in j] for j in df[col]]

    # lemmatization
    lem = WordNetLemmatizer() 
    df[col] = [[lem.lemmatize(i) for i in j] for j in df[col]]
    
    # Transform datetime
    
    ##### Time dependency ana 

    # drop rows with no timestamps
    df = df[df[col1].notna()]

    # transform string in system started to dt object
    df[col1] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') 
                                             for d in df[col1]]

    ######## Time dif between NEXT_RIDE and SYSTEM_STARTED_AT

    # drop Nan in NEXT_RIDE_DT -> BUT remember here, if Nan, it could be the case that people have not taken next ride
    df = df[df[col2].notna()]

    # transform string in system started to dt object
    df[col2] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') 
                                             for d in df[col2]]
    
    return df

# return top n-grams 
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:10]

# dict = id2word, corpus = bow_corpus, texts=nyc_df_clean_text['DESCRIPTION'], start=2, limit=40, step=6

# Calc coherence value for a set of lda models
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
        model = gensim.models.ldamodel.LdaModel(corpus= corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# generate n_gram visuals 
# df_review is pd series with cleaned col of reviews
# n is number of grams 

def viz_n_gram(df_review, n):
    
    df_review = df_review.apply(lambda x: ' '.join(map(str,x)))
    
    top_tri_bigrams = get_top_ngram(df_review,n)[:30]
    x, y = map(list,zip(*top_tri_bigrams)) 

    plt.figure(figsize=(15,10))
    sns.set(font_scale=1.5)
    a = sns.barplot(x=y,y=x, palette = 'PuBu_r')
    # a.axes.set_title("Frequency of 3-Word Keywords",fontsize=20)
    a.set_xlabel("Count",fontsize=20)
    a.set_ylabel("Keyword",fontsize=20)
    
    return None

def tpc_sel_multi(df, tpc_ar, col):
    ### TPC 5: cell phone holder
    
    l = len(tpc_ar)
    
    p = 0
    
    tpc_df = pd.DataFrame(columns = df.columns)

    for i in range(len(df)):
    
        current_doc = df[col].iloc[i]
    
        for j in range(len(current_doc)):
        
            if current_doc[j] in tpc_ar:
                
                p = p + 1
                
            else:
                
                continue
                
        if l == p:
            
            tpc_df = tpc_df.append(df.iloc[i])
            
            p = 0
        else:
            
            p = 0
        
    return tpc_df

# get df with specific words in review
# man kann funktion nochmal umbauen indem in jedem Dokument komplette liste enthalten sein muss 

def tpc_sel(df, tpc_ar, col):
    ### TPC 5: cell phone holder
    tpc_df = pd.DataFrame(columns = df.columns)

    for i in range(len(df)):
    
        current_doc = df[col].iloc[i]
    
        for j in range(len(current_doc)):
        
            if current_doc[j] in tpc_ar:
                
                tpc_df = tpc_df.append(df.iloc[i])
                
                break
                
            else:
                
                continue
    return tpc_df

# extract noise from data (array with length less or equal than length)

def ex_noise_text(df, col, length):
    
    l_ges = len(df)
    
    k = 0
    
    reduced_df = pd.DataFrame(columns = df.columns)
    
    for i in range(len(df)):
    
        current_doc = df[col].iloc[i]
        
        l = len(current_doc)
        
        if l <= length:
            
            k = k + 1
            
        else:
            
            reduced_df = reduced_df.append(df.iloc[i])
            
        
    return k / l_ges, reduced_df


################################# IN THE MAKING 
# create array without specific words/topics
##### hier stimmt was nicht

def create_rest_ar(df, tpc_ar, col):
    
    # counter if word was found in doc
    p = 0
    
    rest_df = pd.DataFrame(columns = df.columns)
#len(df)
    for i in range(5000):
    
        current_doc = df[col].iloc[i]
    
        for j in range(len(current_doc)):
        
        # wenn eins der topics (wörter) enthalten ist, wird die scheisse geskipped
            if current_doc[j] in tpc_ar:
                # word is in doc
                p = p + 1
                break
        # wenn in doc wort nicht enthalten ist, wird row in rest array aufgenommen
            else:
                continue
                
        if p == 1:
            continue
        # word is not in doc
        else:
            rest_df = rest_df.append(df.iloc[i])
                
    return rest_df

# x is array of strings with issues/words/topics/etc.
# y is the axis for bar values
# x_lab is string for x-axis
# y_lab is string for y-axis
def gen_bar_char(x, y, x_lab, y_lab):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    plt.xlabel(x_lab, fontsize=14)
    plt.xticks(rotation = 90) 
    plt.ylabel(y_lab, fontsize=14)
    
    ax.bar(x,y)
    plt.show()
    
    return None

# x is array of strings with issues/words/topics/etc.
# y is the axis for bar values
# x_lab is string for x-axis
# y_lab is string for y-axis
def gen_lin_char(x, y, x_lab, y_lab):
    
    plt.plot(x, y, color='blue', marker='o')
    
    plt.xlabel(x_lab, fontsize=14)
    plt.xticks(rotation = 90) 
    plt.ylabel(y_lab, fontsize=14)
    
    plt.grid(True)
    plt.show()
    
    return None

# LDA model with parameters 
########### apply LDA #################

# df with a cleaned text col (lemmatized, tokenized)
# col column name of doc/text data 
# n is number of topics 
# except n all parameters are set in the function 

def run_lda(df, col, n):
    
    id2word = gensim.corpora.Dictionary(df[col])
    # create dict with all words
    dictionary = gensim.corpora.Dictionary(df[col])
    # create text corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in df[col]]

    # Build simple LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus= bow_corpus,
                                               id2word=id2word,
                                               num_topics=n, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    return lda_model

# Create word cloud for genism lda model
# define stopwords from library
# standardized format for visuals
# more colors: 'mcolors.XKCD_COLORS'
def viz_word_cloud(lda_model):
    stop_words = list(stopwords.words('english'))

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  
    
    # generate Wordcloud objects 
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    # extract topics from lda
    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)
    
    # plot word clouds for each topic
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    
    return None
