
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')       
nltk.download('punkt')

data_train = pd.read_excel('train.xlsx', names=["label", "email"])      #import train file
data_train=data_train.iloc[np.random.permutation(data_train.index)].reset_index(drop=True)  #shuffling it

data_test = pd.read_excel('test.xlsx', names=["label", "email"])  #import test file

#data_validation = data_train.iloc[3000:]    #taking the last 657 rows for validation

train_val = data_train            
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'



#adding feature engineering techniques

def remove_links(email):
    '''Takes a string and removes web links from it'''
    email = re.sub(r'http\S+', '', email) # remove http links
    email = re.sub(r'bit.ly/\S+', '', email) # rempve bitly links
    email = email.strip('[link]') # remove [links]
    return email

def remove_users(email):
    '''Takes a string and removes user information'''
    email = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', email) # remove retweet
    email = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', email) # remove tweeted at
    return email


def preprocess(sent):
    sent = remove_users(sent)
    sent = remove_links(sent)
    sent = sent.lower() # lower case
    sent = re.sub('['+my_punctuation + ']+', ' ', sent) # strip punctuation
    sent = re.sub('\s+', ' ', sent) #remove double spacing
    sent = re.sub('([0-9]+)', '', sent) # remove numbers
    sent_token_list = [word for word in sent.split(' ')]
    sent = ' '.join(sent_token_list)
    return sent


# count number of characters 
def count_chars(text):
    return len(text)

# count number of words 
def count_words(text):
    return len(text.split())

# count number of capital characters
def count_capital_chars(text):
    count=0
    for i in text:
        if i.isupper():
            count+=1
    return count

# count number of capital words
def count_capital_words(text):
    return sum(map(str.isupper,text.split()))

# count number of punctuations
def count_punctuations(text):
    punctuations='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    d=dict()
    for i in punctuations:
        d[str(i)+' count']=text.count(i)
    return d

# count number of words in quotes
def count_words_in_quotes(text):
    x = re.findall("\'.\'|\".\"", text)
    count=0
    if x is None:
        return 0
    else:
        for i in x:
            t=i[1:-1]
            count+=count_words(t)
        return count
    
# count number of sentences
def count_sent(text):
    return len(nltk.sent_tokenize(text))

# calculate average word length
def avg_word_len(char_cnt,word_cnt):
    return char_cnt/word_cnt

# calculate average sentence length
def avg_sent_len(word_cnt,sent_cnt):
    return word_cnt/sent_cnt

# count number of unique words 
def count_unique_words(text):
    return len(set(text.split()))
            
# words vs unique feature
def words_vs_unique(words,unique):
    return unique/words

# count of mentions
def count_mentions(text):
    x = re.findall(r'(\@\w[A-Za-z0-9]*)', text)
    return len(x)

# count of stopwords
def count_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

# stopwords vs words
def stopwords_vs_words(stopwords_cnt,text):
    return stopwords_cnt/len(word_tokenize(text))


train_val = train_val.dropna(subset=['email'])   #drop float /nan rows in email column
train_val = train_val.reset_index(drop=True)


def create_features(data):
    data['char_count'] = data["email"].apply(lambda x:count_chars(x))
    data['word_count'] = data["email"].apply(lambda x:count_words(x))
    data['sent_count'] = data["email"].apply(lambda x:count_sent(x))
    data['capital_char_count'] = data["email"].apply(lambda x:count_capital_chars(x))
    data['capital_word_count'] = data["email"].apply(lambda x:count_capital_words(x))
    data['quoted_word_count'] = data["email"].apply(lambda x:count_words_in_quotes(x))
    data['stopword_count'] = data["email"].apply(lambda x:count_stopwords(x))
    data['unique_word_count'] = data["email"].apply(lambda x:count_unique_words(x))
    data['mention_count'] = data["email"].apply(lambda x:count_mentions(x))
    data['punct_count'] = data["email"].apply(lambda x:count_punctuations(x))
    data['avg_wordlength']=data['char_count']/data['word_count']
    data['avg_sentlength']=data['word_count']/data['sent_count']
    data['unique_vs_words']=data['unique_word_count']/data['word_count']
    data['stopwords_vs_words']=data['stopword_count']/data['word_count']

    return data

train_val = create_features(train_val)   #create features in train
data_test = create_features(data_test)   #create features in test

df_punct= pd.DataFrame(list(train_val.punct_count))
test_punct= pd.DataFrame(list(data_test.punct_count))

# Mearning punctuation DataFrame with main DataFrame
train_val=pd.merge(train_val,df_punct,left_index=True, right_index=True)
data_test=pd.merge(data_test,test_punct,left_index=True, right_index=True)

# We can drop "punct_count" column from both df and test DataFrame
train_val.drop(columns=['punct_count'],inplace=True)
data_test.drop(columns=['punct_count'],inplace=True)

train_val['email']=train_val['email'].apply(lambda x: preprocess(x))
data_test['email']=data_test['email'].apply(lambda x: preprocess(x))

train_val['email'] = train_val['email'].astype('string')
data_test['email'] = data_test['email'].astype('string')
train_val['email'].fillna('', inplace = True)
data_test['email'].fillna('', inplace = True)
#converting all the data to vectors with important features using TFIDF 
vectorizer            =  TfidfVectorizer()
train_tf_idf_features =  vectorizer.fit_transform(train_val['email']).toarray()
test_tf_idf_features  =  vectorizer.transform(data_test['email']).toarray()
# Converting above list to DataFrame
train_tf_idf          = pd.DataFrame(train_tf_idf_features)
test_tf_idf           = pd.DataFrame(test_tf_idf_features)
# Saparating train and test labels from all features
train_Y               = train_val['label']
test_Y                = data_test['label']
# Listing all features
features = ['char_count', 'word_count', 'sent_count',
       'capital_char_count', 'capital_word_count', 'quoted_word_count',
       'stopword_count', 'unique_word_count', 'mention_count',
       'avg_wordlength', 'avg_sentlength', 'unique_vs_words',
       'stopwords_vs_words', '! count', '" count', '# count', '$ count',
       '% count', '& count', '\' count', '( count', ') count', '* count',
       '+ count', ', count', '- count', '. count', '/ count', ': count',
       '; count', '< count', '= count', '> count', '? count', '@ count',
       '[ count', '\ count', '] count', '^ count', '_ count', '` count',
       '{ count', '| count', '} count', '~ count']
# Finally merging all features with above TF-IDF. 
train = pd.merge(train_tf_idf,train_val[features],left_index=True, right_index=True)
data_test  = pd.merge(test_tf_idf,data_test[features],left_index=True, right_index=True)

X_train, X_test, y_train, y_test = train_test_split(train, train_Y, test_size=0.2, random_state = 42)# Random Forest Classifier
#using Random FOrest as classifier
_RandomForestClassifier = RandomForestClassifier(n_estimators = 1000, min_samples_split = 15, random_state = 42)
_RandomForestClassifier.fit(X_train, y_train)
_RandomForestClassifier_prediction = _RandomForestClassifier.predict(X_test)
val_RandomForestClassifier_prediction = _RandomForestClassifier.predict(data_test)
print("Accuracy => ", round(accuracy_score(_RandomForestClassifier_prediction, y_test)*100, 2))
print("\nRandom Forest Classifier results: \n")
print(classification_report(y_test, _RandomForestClassifier_prediction, target_names = ['Yes', 'No']))
print("Validation Accuracy => ", round(accuracy_score(val_RandomForestClassifier_prediction, test_Y)*100, 2))
print("\nValidation Random Forest Classifier results: \n")
print(classification_report(test_Y, val_RandomForestClassifier_prediction, target_names = ['Yes', 'No']))


