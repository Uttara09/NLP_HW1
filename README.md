# NLP_HW1
Sentiment Analysis of Tweets

Name : Uttara Ravi
Lionmail : ur2135@columbia.edu

The code in hw1.py trains the model to perform sentiment analysis of tweets on twitter. It has four different models each with different features:-
1. Ngram model that uses uni,bi,tri and 4-grams as features
2. Ngram + Lex model that uses the features in model 1 and  Lex features mined from the Sentiment140-Lexicon for uni and bi grams.
3. Ngram + Lex + Encoding model that uses the features in model in addition to the count of hashtags and all-caps words in every tweet.
4. Custom model that uses the features in Model 3 in addition to the count of elongated words and Punctuations (like "soooo", "!!!!") in every tweet.

To train and test my classifier please run it as follows: 

python hw1.py --train <path-to-resources>/data/train.csv --test <path-to-resources>/data/dev.csv --model "Model name" --lexicon_path <path-to-resources>/lexica/

I have used a MultinomialNB classifier for all four models with default parameters. The features are being created in the features.py file and after being added to the dataframe are being returned to hw1.py. Here there is an if condition that fits the data and predicts only if that particular features’ model is provided as an argument while executing the file.
