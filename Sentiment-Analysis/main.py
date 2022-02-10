# %% [markdown]
# # First we define some functions to preprocess the text documents before we do the training.
# 

# %%
import re           
"""
    we need re for basic string pattern 
    like removing non alphabet characters in removeNonAlpha function
"""
import nltk
stop_words_set = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def removeNonAlpha(s: str):
    return re.sub(r'[^A-Za-z\s]', '', s)
"""
    here because to apply three operations (remove stop words, stemming and lemmatizing)
    we need to tokenize the whole string and then concat all tokens at the end. we did all three operations in one functions
    to avoid repeating tokenizing and joining tokens ... 
"""



def removeStopWordsAndStemAndLemmatize(s: str):
    return ' '.join([stemmer.stem(lemmatizer.lemmatize(token)) for token in nltk.tokenize.word_tokenize(s) if not token in stop_words_set])
"""
    In this function we first lemmatize the token. for example it converts "cities" -> "city"
    or some irregular cases like: "mice" -> "mouse"

    Then we stemm the token. somehow it converts some tokens to their root. 
    like: several -> sever   

    And then we add it to the list only if it's not a stop word. then we join all tokens with ' ' and return the string

    Reference for preprocessing: 
        https://github.com/aravinthsci/Text-Preprocessing-in-Python/blob/master/Text_Preprocessing_in_Python.ipynb
"""





def strPreProccess(s: str):
    return removeStopWordsAndStemAndLemmatize(removeNonAlpha(s.strip().lower()))
"""
A funciton that does all the job at once
"""


# %% [markdown]
# # First the "SandersPosNeg" dataset

# %% [markdown]
# ## Now we read the data from the file.

# %%

import pandas as pd

data = pd.read_csv("./SandersPosNeg.csv", sep='\t', header=None) # we have no header in dataset so set header=None
data # a preview of the dataset

# %% [markdown]
# ## As you see the texts need to be proccessed and get ready for training(i checked and saw it increases the accuracy about 2 to 3 percent)

# %% [markdown]
# ### For a better understanding of dataset we rename the columns
# #### And then apply the preprocessing to the tweet text column 

# %%
data.rename(columns={0: "label", 1: "tweet text"}, inplace=True)
data['tweet text'] = data['tweet text'].apply(strPreProccess)
data # a preview of data

# %% [markdown]
# ## As you can see the data is now ready to train

# %% [markdown]
# ### First we import some neccesary libraries

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit, cross_val_score
import sklearn.naive_bayes as NB

# %% [markdown]
# ### now before we use a predicter we need to convert the text to numerical values to work on
# #### To do this we use Tf-idf vectorization

# %%
vectorizer = tfidf_vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['tweet text'])
Y = data['label']



# %% [markdown]
# ### Then we create an Multinomial Naive Bayse model instance to give to the cross_val_score function so it fits the data into given model and measure our accuracy using 10-fold-cross-validation method. 
# 
# #### (Obviously the X will be the vectors(converted from tweet texts) 
# #### and Y will be labels (0 for negative and 4 for positive))

# %%
model = NB.MultinomialNB()
NB_result = cross_val_score(model, X, Y, cv=ShuffleSplit(10, test_size=0.2, random_state=0)).mean()
print(f'NB: {NB_result.mean()*100}')

# %% [markdown]
# # Now we do the same thing with the OMD dataset. 
# ### Except for some tiny details. for example we need to use "mac_roman" encoding.
# ### Also the csv file is seperated by camma which also appeare in tweet texts so setting the sep=',' would cause a mess!
# ### To handle the situation first we read the whole line in a step.(we will have one column)
# ### Next step we split it by the first camma we see(it seperates the label and tweet text)
# 

# %%
data = pd.read_csv("./OMD.csv", header=None, sep='\t', encoding='mac_roman')

def getLabel(s: str) -> int:
    return int(s[:s.find(',')])

def getTweet(s: str) -> str:
    return s[s.find(','):]

data['label'] = data[0].apply(getLabel)
data['tweet text'] = data[0].apply(getTweet)
del data[0]
data


# %% [markdown]
# # And from now on everything is the same as SandersPosNeg dataset

# %%
data["tweet text"] = data['tweet text'].apply(strPreProccess)
data

# %% [markdown]
# ## One tiny point! we set the alpha parameter here in our MultinomialNB to 0.49 so it gives us about 2 percent more accuracy!

# %%
vectorizer = tfidf_vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['tweet text'])
Y = data['label']

model = NB.MultinomialNB(alpha=0.49)
NB_result = cross_val_score(model, X, Y, cv=ShuffleSplit(10, test_size=0.2, random_state=0)).mean()
print(f'NB: {NB_result.mean()*100}')


