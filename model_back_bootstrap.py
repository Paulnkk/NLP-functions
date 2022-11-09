# runs = number of boostrap runs
# n = number of boostrap samples 
# cv_ar = array with cv arrays
# nyc_df_clean_text = prepared and tokenized nyc data (whole dataset)
# topics = topics from modeling approach 
# df_boo = boostrap dataset

dictionary_easy = gensim.corpora.Dictionary(nyc_df_clean_text['DESCRIPTION'])
bow_corpus_easy = [dictionary_easy.doc2bow(doc) for doc in nyc_df_clean_text['DESCRIPTION']]

cv_ar = []
n = 5000
runs = 20
rdm_state = 1234

for i in range(runs):
    
    df_boo = nyc_df_clean_text.sample(n, random_state = rdm_state)
    
    #dictionary_easy = gensim.corpora.Dictionary(df_boo['DESCRIPTION'])
    
    #bow_corpus_easy = [dictionary_easy.doc2bow(doc) for doc in df_boo['DESCRIPTION']]
    
    cm_gsdmm = CoherenceModel(topics = topics, 
                              dictionary = dictionary_easy, 
                              corpus = bow_corpus_easy, 
                              texts = df_boo['DESCRIPTION'], 
                              coherence = 'c_v', topn = 50)
    
    coherence_gsdmm = cm_gsdmm.get_coherence()  
    
    cv_ar.append(coherence_gsdmm)
    
    rdm_state = rdm_state + 1
