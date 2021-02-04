Homework - 1
Readme File


Name : Uttara Ravi
Lionmail : ur2135@columbia.edu


To train and test my classifier please run it as follows: 
1.
python hw1.py --train <path-to-resources>/data/train.csv --test <path-to-resources>/data/dev.csv --model "Model name" --lexicon_path <path-to-resources>/lexica/


2. Please make sure that the directory lexica contains Sentiment140-Lexicon which contains the files Emoticon-unigrams.txt and Emoticon-bigrams.txt. Make sure to not miss the “/” at the end of the path to lexica.


3. I have used a MultinomialNB classifier for all four models with default parameters. The custom feature I chose to implement is a feature that counts the number of elongated words like “sooo”, “wowwww” along with repeated punctuation like “!!!!!” and “????”.


4. The features are being created in the features.py file and after being added to the dataframe are being returned to hw1.py. Here there is an if condition that fits the data and predicts only if that particular features’ model is provided as an argument while executing the file.


5. Another point to note is that I am incrementally creating my model after each set of features. For example if you choose Ngram as the model I will only create ngram features and train my classifier using that. If you choose Ngram+Lex, Ngram and lex features will be created and then a classifier will be trained on this dataset and predict values, and so on.
(If you want to see all the features being created and want to evaluate it on how time intensive it is, I suggest you pick the custom model.) 


6. Limitation : My code can only take csv files as input for training and test datasets because I have used the read_csv function by pandas to create a dataframe.
