#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:44:17 2021

@author: uttararavi
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from tqdm import tqdm
import nltk
from re import search
from nltk.tokenize import word_tokenize
def get_ngrams_features(X_train, X_dev): 
    print('Inside get_ngrams_features')
    ### Ignoring all words that occur in more than 98 percent of the documents, 
    ### and less than 0.2 percent of the documents
    ### Taking only the 1000 best ngrams that have been created
    ng_vectorizer = CountVectorizer(min_df=0.002, max_df=0.98,ngram_range=(1, 4), max_features=1000) 
    X_train = ng_vectorizer.fit_transform(X_train)
    X_dev = ng_vectorizer.transform(X_dev)  

    return X_train, X_dev

def get_lex_features(tweets, lex_emoticons_unigrams, lex_emoticons_bigrams):
    print('Inside get_lex_features')
    
    ### get uni gram features
    lex_uni_features = get_lex_uni_features(tweets, lex_emoticons_unigrams)  
    
    #### get bi gram features
    lex_bi_features = get_lex_bi_features(tweets,lex_emoticons_bigrams)
    
    ### concatenate unigram and bigram features
    lex_features = pd.concat([lex_uni_features,lex_bi_features], axis=1)
    
    return lex_features

def get_enc_features(tweets):
    print('Inside get_enc_features')
    enc_features = pd.DataFrame()
    n = 0
    for ind in tqdm(tweets.index):
        tweet = tweets['tweet_tokens'][ind].split()
        hashtag_count = 0
        all_caps_count = 0
        ### getting the total number of all caps words and total number of hashtags in the tweet
        for word in tweet:
            if '#' in word:
                hashtag_count += 1
            if word.isupper():
                all_caps_count += 1
        
        #### add row to enc_features
        if len(enc_features) > 0:
                enc_features.loc[len(enc_features)] = [hashtag_count,all_caps_count]
        else:
            data = [[hashtag_count,all_caps_count]]
            enc_features = pd.DataFrame(data, columns = ['hashtag_count','all_caps_count'])
        
        n += 1
             
    return enc_features

def get_custom_features(tweets,lex_emoticons_unigrams):
    print('Inside get_custom_features')
    custom_features = pd.DataFrame()
    n = 0
    
    #### calculate number of elongated words in a tweet
    for ind in tqdm(tweets.index):
        tweet = tweets['tweet_tokens'][ind].split()
        regex = re.compile(r"(.)\1{2}")
        ### this gets the counts of all occurances of elongated words like "soooo"
        ### It also includes repeated punctuations like "!!!!"
        elong_count = sum([1 for word in tweet if regex.search(word)])
       
        n += 1
        #### add row to custom_features
        if len(custom_features) > 0:
            custom_features.loc[len(custom_features)] = [elong_count]
        else:
            data = [[elong_count]]
            custom_features = pd.DataFrame(data, columns = ['elong_count'])          
    
    return custom_features 
    

    
def get_lex_uni_features(tweets,lex_emoticons_unigrams):
    print('Inside get_lex_uni_features')
    lex_uni_features = pd.DataFrame()
    n = 0
    
    for ind in tqdm(tweets.index):
        tweet = tweets['tweet_tokens'][ind].split()
        
        f1,f5 = 0,0  # 1. total count of all the unigrams in the tweet with pos and neg polarity respectively
        f2,f6 = 0 ,0 # 2. total score of the unigrams in pos and neg polarity respectively
        tweet_scores_pos = [] 
        tweet_scores_neg = []
        for word in tweet:
            word_lower = word.lower()
            if word_lower in lex_emoticons_unigrams.index:
                if lex_emoticons_unigrams['score'].loc[word_lower] > 0:
                    f1 += 1
                    tweet_scores_pos.append(lex_emoticons_unigrams['score'].loc[word_lower])
                    f2 += lex_emoticons_unigrams['score'].loc[word_lower]
                if lex_emoticons_unigrams['score'].loc[word_lower] < 0:
                    f5 += 1
                    tweet_scores_neg.append(-1*lex_emoticons_unigrams['score'].loc[word_lower])
                    f6 += (-1*lex_emoticons_unigrams['score'].loc[word_lower])
                
                
        if tweet_scores_pos:
            f3 = max(tweet_scores_pos)  # 3. max score of a pos polarity unigram in the tweet
            f4 = tweet_scores_pos[-1]   # 4. score of the last unigram in the tweet with positive polarity
        else:
            f3 = 0
            f4 = 0
        if tweet_scores_neg:
            f7 = max(tweet_scores_neg)  # 3. max score of a neg polarity unigram in the tweet
            f8 = tweet_scores_neg[-1]   # 4. score of the last unigram in the tweet with negative polarity
        else:
            f7 = 0
            f8 = 0
        
        n += 1
        #### add row to lex_features
        if len(lex_uni_features) > 0:
            lex_uni_features.loc[len(lex_uni_features)] = [f1,f2,f3,f4,f5,f6,f7,f8]
        else:
            data = [[f1,f2,f3,f4,f5,f6,f7,f8]]
            lex_uni_features = pd.DataFrame(data, columns = ['f1','f2','f3','f4','f5','f6','f7','f8']) 
            

    return lex_uni_features 
    

def get_lex_bi_features(tweets,lex_emoticons_bigrams):
    print('Inside get_lex_bi_features')
    lex_bi_features = pd.DataFrame()
    n = 0
    
    for ind in tqdm(tweets.index):
        tweet = tweets['tweet_tokens'][ind].split()
        bigram_list = nltk.bigrams(tweet)
        f1,f5 = 0,0  # 1. total count of all the bigrams in the tweet with pos and neg polarity respectively
        f2,f6 = 0 ,0 # 2. total score for the bigrams with pos and neg polarity respectively
        tweet_scores_pos = []
        tweet_scores_neg = []
        for bigram in bigram_list:
            bigram_lower = bigram[0].lower() + " " + bigram[1].lower()
            
            if bigram_lower in lex_emoticons_bigrams.index:
                if lex_emoticons_bigrams['score'].loc[bigram_lower] > 0:
                    f1 += 1
                    tweet_scores_pos.append(lex_emoticons_bigrams['score'].loc[bigram_lower])
                    f2 += lex_emoticons_bigrams['score'].loc[bigram_lower]
                if lex_emoticons_bigrams['score'].loc[bigram_lower] < 0:
                    f5 += 1
                    tweet_scores_neg.append(-1*lex_emoticons_bigrams['score'].loc[bigram_lower])
                    f6 += (-1*lex_emoticons_bigrams['score'].loc[bigram_lower])
                
                
        if tweet_scores_pos:
            f3 = max(tweet_scores_pos)  # 3. max score of a positive bigram in the tweet
            f4 = tweet_scores_pos[-1]   # 4. score of the last bigram in the tweet with positive polarity
        else:
            f3 = 0
            f4 = 0
        if tweet_scores_neg:
            f7 = max(tweet_scores_neg)  # 3. max score of a negative bigram in the tweet
            f8 = tweet_scores_neg[-1]   # 4. score of the last bigram in the tweet with negative polarity
        else:
            f7 = 0
            f8 = 0
        
        n += 1

        #### add row to lex_features
        if len(lex_bi_features) > 0:
            lex_bi_features.loc[len(lex_bi_features)] = [f1,f2,f3,f4,f5,f6,f7,f8]
        else:
            data = [[f1,f2,f3,f4,f5,f6,f7,f8]]
            lex_bi_features = pd.DataFrame(data, columns = ['f1','f2','f3','f4','f5','f6','f7','f8']) 
          
    return lex_bi_features 
    
    

    
    
    