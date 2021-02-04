import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import csv
import features
import sys

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()
  
    #### Load data
    tweets_train = pd.read_csv(args.train)   
    tweets_dev = pd.read_csv(args.test)
    #### replace the classfication column with 0,1,2
    tweets_train['label'] = tweets_train['label'].replace(['negative','positive','neutral','objective'],[0,1,2,2])
    tweets_dev['label'] = tweets_dev['label'].replace(['negative','positive','neutral','objective'],[0,1,2,2])
    target_names = ['0','1','2']
    
    # print(tweets_train.groupby('label').label.count())

    #### Seperate and get list of tweets 
    y_train = tweets_train.label
    X_train = tweets_train['tweet_tokens']
    
    y_dev = tweets_dev.label
    X_dev = tweets_dev['tweet_tokens']
    
    ### Model 1 : ngrams 
    ### Getting all Ngram features
    X_train, X_dev = features.get_ngrams_features(X_train, X_dev)
    
    if args.model == "Ngram":
        print('Model 1 : Ngrams')
        #### Pick a classifier
        clf = MultinomialNB()
        #### Fit the classifier
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_dev)
        print(metrics.classification_report(y_dev, y_predicted,
                                            target_names=target_names))
        
        #exiting because we don't want to train for other features
        sys.exit(0)
    
    
    ### Model 2 : ngrams + lex
    #### create a datframe of the lex-unigram file
    lex_uni_file = args.lexicon_path + "Sentiment140-Lexicon/Emoticon-unigrams.txt"
    lex_bi_file = args.lexicon_path + "Sentiment140-Lexicon/Emoticon-bigrams.txt"

    lex_emoticons_unigrams = pd.read_table(lex_uni_file, delim_whitespace=True, names=('term', 'score','npos','nneg'))
    ### dropping npos and nneg as it isn't being used elsewhere
    lex_emoticons_unigrams.drop(columns=['npos','nneg'])
    lex_emoticons_unigrams.set_index("term", drop=True, inplace=True)


    #### create a datframe of the lex-bigram file
    lex_emoticons_bigrams = pd.read_csv(lex_bi_file, sep = "\t", \
                                        names=['term', 'score','npos','nneg'], quoting=csv.QUOTE_NONE, error_bad_lines = False)
    ### dropping npos and nneg as it isn't being used elsewhere
    lex_emoticons_bigrams.drop(columns=['npos','nneg'])
    lex_emoticons_bigrams.set_index("term", drop=True, inplace=True)
    
    #### get lex features
    lex_features_train = features.get_lex_features(tweets_train,lex_emoticons_unigrams,lex_emoticons_bigrams)
    ### append lex features to our existing feature set
    X_train = pd.DataFrame(X_train.toarray())
    X_train = pd.concat([X_train,lex_features_train], axis = 1)
    
    lex_features_dev = features.get_lex_features(tweets_dev,lex_emoticons_unigrams,lex_emoticons_bigrams)
    ### append lex features to our existing feature set
    X_dev = pd.DataFrame(X_dev.toarray())   
    X_dev = pd.concat([X_dev,lex_features_dev], axis = 1)
    
    if args.model == "Ngram+Lex":
        print('Model 2 : Ngrams+Lex')
        #### Pick a classifier
        clf = MultinomialNB()
        #### Fit the classifier
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_dev)
        print(metrics.classification_report(y_dev, y_predicted,
                                        target_names=target_names))
        #exiting because we don't want to train for other features
        sys.exit(0)
    
    ### Model 3 : ngrams + lex + enc 
    ### get encoding features
    enc_features_train = features.get_enc_features(tweets_train)
    enc_features_dev = features.get_enc_features(tweets_dev)
    
    ### append encoding features to our existing feature set
    X_train = pd.concat([X_train, enc_features_train], axis = 1)
    X_dev = pd.concat([X_dev, enc_features_dev], axis = 1)
    
    if args.model == "Ngram+Lex+Enc":
        print('Model 3 : Ngrams+Lex+Enc')
        #### Pick a classifier
        clf = MultinomialNB()
        #### Fit the classifier
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_dev)
        print(metrics.classification_report(y_dev, y_predicted,
                                        target_names=target_names))
        #exiting because we don't want to train for other features
        sys.exit(0)
    
    
    ### Model 4 : ngrams + lex + enc + custom
    
    ### get custom features
    custom_features_train = features.get_custom_features(tweets_train,lex_emoticons_unigrams)
    custom_features_dev = features.get_custom_features(tweets_dev,lex_emoticons_unigrams)
    
    ### append encoding features to our existing feature set
    X_train = pd.concat([X_train,custom_features_train], axis = 1)
    X_dev = pd.concat([X_dev,custom_features_dev], axis = 1)
    
    
    
    if args.model == "Custom":
        print('Model 4 : Custom')
        #### Pick a classifier
        clf = MultinomialNB()
        #### Fit the classifier
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_dev)
        print(metrics.classification_report(y_dev, y_predicted,
                                        target_names=target_names))
    
    ### finally exit from our main function
    sys.exit(0)
    
   
    





    