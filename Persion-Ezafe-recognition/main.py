# %% [markdown]
# # First we import pandas to read the data set
# ## Then we define some constants that will be our column names later

# %%
import pandas as pd
TOKEN = 'token'
POS_TAG = 'pos_tag'
EZAFE_TAG = 'ezafe_tag'
NEXT_TOKEN_POS = 'next_token_pos'
PREV_TOKE_POS = 'prev_token_pos'
LAST_LETTER = 'last_letter'
TOKEN_LENGTH = 'token_length'
IS_VERB = 'is_verb'
HAS_AA_BAKOLAAH ='has_aa_baakolaah'
HAS_AA_BIKOLAAH = 'has_aa_bikolaah'
HAS_TARIN = 'has_tarin'
HAS_TAR = 'has_tar'
HAS_VAAV = 'has_vaav'
HAS_HE = 'has_he'

# %% [markdown]
# ## Now we read the data. And it also has header
# ### Since we didn't set 'header=None' pd won't be recognized as a pd.DataFrame so we declare it as a pd.DataFrame(just for syntax highlighting and suggestions)
# ### Pay attention that there's an extra useless column(named 'Unnamed: 0') so we drop it
# ### Then we get a first preview of our raw data

# %%
df = pd.read_csv("./dataset/updated_bijankhan_corpus.csv")
df: pd.DataFrame

del df['Unnamed: 0']
df.head(20)

# %% [markdown]
# ## As you can see we have sentences coming after together having their words in each row. And there is a delimiter between each sentence as well.

# %% [markdown]
# ## Now lets take a look at each column details

# %%
df[TOKEN].value_counts()

# %%
df[POS_TAG].value_counts()

# %% [markdown]
# ### Note that we hav 482 defferent types of pos_tags so probably this can be our best feature!

# %% [markdown]
# # Here we define some functions to analyze last letters of words in 'token' columns

# %%
def hasSuffixABaKolaah(s: str) -> bool:
    return s.endswith('آ')
def hasSuffixABiKolaah(s: str) -> bool:
    return s.endswith('ا')
def hasSuffixTarin(s: str) -> bool:
    return s.endswith('ترین')
def hasSuffixTar(s: str) -> bool:
    return s.endswith('تر')
def hasSuffixVaav(s: str) -> bool:
    return s.endswith('و')
def hasSuffixHe(s: str) -> bool:
    return s.endswith('ه')


# %% [markdown]
# ## For each special suffix we add a new boolean column where it will be True IFF the token in that row has the corresponding suffix
# ### Let's look at the data again 

# %%
df[HAS_AA_BAKOLAAH] = df[TOKEN].apply(hasSuffixABaKolaah)
df[HAS_AA_BIKOLAAH] = df[TOKEN].apply(hasSuffixABiKolaah)
df[HAS_TARIN] = df[TOKEN].apply(hasSuffixTarin)
df[HAS_TAR] = df[TOKEN].apply(hasSuffixTar)
df[HAS_VAAV] = df[TOKEN].apply(hasSuffixVaav)
df[HAS_HE] = df[TOKEN].apply(hasSuffixHe)
df

# %% [markdown]
# # Here we're making sure that some rules are respected

# %% [markdown]
# ### first check how many of the rows in which the token has آ doesn't have EZAFE

# %%
df[df[HAS_AA_BAKOLAAH]][EZAFE_TAG].hist()
df[df[HAS_AA_BAKOLAAH]][EZAFE_TAG].value_counts()

# %% [markdown]
# ### Suprisingly there is 5! let's see what cases we have

# %%
df[(df[HAS_AA_BAKOLAAH]) & df[EZAFE_TAG]].head(20)

# %% [markdown]
# #### although 5 is not a big number to affect the model. But it would be better if they had 'ی' at the end

# %% [markdown]
# ### Let's look at 'ا':
# #### here we have more! 378 cases!

# %%
df[(df[HAS_AA_BIKOLAAH])][EZAFE_TAG].hist()
df[(df[HAS_AA_BIKOLAAH])][EZAFE_TAG].value_counts()

# %% [markdown]
# ### Let's look at the exact cases

# %%
df[(df[HAS_AA_BIKOLAAH]) & df[EZAFE_TAG]].head(20)

# %% [markdown]
# #### Most of these must have a 'ی' at the end!

# %% [markdown]
# ### Now ترین and تر can get EZAFE sometimes. it's not wrong so we just check the numbers and pass

# %%
df[(df[HAS_TARIN])][EZAFE_TAG].hist()
df[(df[HAS_TARIN])][EZAFE_TAG].value_counts()

# %%
df[(df[HAS_TAR])][EZAFE_TAG].hist()
df[(df[HAS_TAR])][EZAFE_TAG].value_counts()

# %%
del df[HAS_TARIN]
del df[HAS_TAR]

# %% [markdown]
# ### Now let's take a look at tokens having 'و' as their suffix. We have 140k of them and 1.8k has EZAFE! let's see them in details...

# %%
df[(df[HAS_VAAV])][EZAFE_TAG].hist()
df[(df[HAS_VAAV])][EZAFE_TAG].value_counts()

# %%
df[(df[HAS_VAAV]) & df[EZAFE_TAG]]

# %% [markdown]
# #### These cases seem rational too and can't help us mush. because the sound 'v' as well(not just sound 'oo')

# %% [markdown]
# ### Now let's the tokens with 'ا'. We have 254k of them and suprisingly theres 64215 of them getting EZAFE!!!

# %% [markdown]
# #### Let's look in details

# %%
df[(df[HAS_HE])][EZAFE_TAG].hist()
df[(df[HAS_HE])][EZAFE_TAG].value_counts()

# %%
df[(df[HAS_HE]) & df[EZAFE_TAG]]

# %% [markdown]
# #### This is too much! we need to fix them. we need to add a letter 'ی' to these tokens

# %% [markdown]
# ### We need to append a letter 'ی' to those tokens having some special suffix(last letter) AND EZAFE! we have 2 conditions. consider each condition 1. we construct a new column each row is a number between 0 and 2. showing how many of those conditions it holds.(2s are our target. one for having special last letter and one for having EZAFE)

# %% [markdown]
# ### In here we difine two functions. one just determines if the word has some of those special suffixes
# ### The other replaces '2'(the row having both conditions) with 'ی' and "empty string" for others.

# %%
def hasSpecialLastLetter(s: str) -> str:
    return (s.endswith('ه') and not s.endswith('اه')) or s.endswith('ا') or s.endswith('آ')
    
def replaceYeWithTwo(num: int) -> str:
    return '‌ی' if num==2 else ''

# %% [markdown]
# #### Now here we build and and concat it to the end of the 'token' columns

# %%
df['temp'] = df[TOKEN].apply(hasSpecialLastLetter)

'''
    the temp column has 1 wherever the corresponding has special last letter
'''

df['temp'] = df['temp'] + df[EZAFE_TAG]
'''
This operations apply an logical AND between those 2 conditions. we will have '2' IFF the token has EZAFE AND special last letter
Actually let's take a look at it
'''

df['temp'].value_counts()
df['temp'].hist()


# %% [markdown]
# ### Just as expected! 57074 BAD cases(tokens) that need a letter 'ی' at their end

# %%
df[TOKEN] = df[TOKEN] + df['temp'].apply(replaceYeWithTwo)
df[df['temp']==2]

# %% [markdown]
# ### As you can see now it's correct!

# %% [markdown]
# ## Now for Another pre processing we need to insert a new column 'last_letter' to do this we need to define the following function:

# %%
special_suffixes = {
                    # 'تر',
                    # 'ترین',
                    'ا',
                    'آ',
                    # 'و',
                    'ه',
                }
def classifyLastLetter(s: str) -> int:
    if s[-1] in special_suffixes:
        return 'special'
    if s[-1].isalpha():
        return s[-1]
    if s[-1].isnumeric():
        return 'numeric'
    else:
        return 'other'
    

# %% [markdown]
# ### And create 2 new columns. suprisingly the length of the token will be helpful!

# %%
df[LAST_LETTER] = df[TOKEN].apply(classifyLastLetter)
df[TOKEN_LENGTH] = df[TOKEN].apply(lambda s: len(s))

# %%
df[TOKEN_LENGTH].hist(bins=30)
df[TOKEN_LENGTH]

# %% [markdown]
# ### we know that the verbs won't ever have EZAFE. so we create a boolean column for that as well.

# %%
df[IS_VERB] = df[POS_TAG].apply(lambda s: str(s).startswith('V_')).astype(int)
df[IS_VERB].hist()

# %% [markdown]
# ## Now since we can't work with string data we need to convert them to numeric data using a label encoder.

# %% [markdown]
# ##### And a preview of our data frame

# %%
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df[POS_TAG] = label_encoder.fit_transform(df[POS_TAG])
df[LAST_LETTER] = label_encoder.fit_transform(df[LAST_LETTER])
df[TOKEN] = label_encoder.fit_transform(df[TOKEN])


# %%
df[POS_TAG].hist(bins=len(df[POS_TAG].unique()))
df[POS_TAG]

# %%
df[LAST_LETTER].hist(bins=80)
df[LAST_LETTER]

# %% [markdown]
# #### But an important thing is that the pos_tag of the next and [maybe] the previous token can be a greate help
# #### so we make a shift to top and down to have them.
# #### we also set the NaN values (generated by shifts) to zero

# %%
df[PREV_TOKE_POS] = df[POS_TAG].shift(periods=1)
df[NEXT_TOKEN_POS] = df[POS_TAG].shift(periods=-1)
df[NEXT_TOKEN_POS][len(df[NEXT_TOKEN_POS])-1] = 0
df[PREV_TOKE_POS][0] = 0
df

# %% [markdown]
# # Now our data is pre processed and ready to treain.
# ### Since we will have to try different models with defferent parameters again and again we define these simple functions

# %%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

'''
    This simple function takes a model 'model' and the list of columns 'X_columns' to use as featues.
    Then fit the data througth model using cross validation method and prints out the result.
'''
def runCrossValOnModel(model, X_columns: list):
    X = df[X_columns]
    Y = df[EZAFE_TAG] 
    cv = ShuffleSplit(10, test_size=0.2, random_state=0)
    print(f"result: {cross_val_score(model, X, Y, cv=cv).mean()*100}")
    

'''
This one take a model 'model' and the column names 'Xs' and randomly split it into train and test parts and fit the train part to the model.
Then using that model predict the test part and calculate Accuracy, Precision and Recall.
we have defferent parameters for precision and recall so we try them all.
'''
def runRandomSplitOnModel(model, Xs):
    X = df[Xs]
    Y = df[EZAFE_TAG]
    X_train, X_test, T_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train, T_train)
    y_predict = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(Y_test, y_predict)}')

    for parameter in ['binary', None, 'micro', 'macro', 'weighted']:
        print(f'Precision [avg: {parameter}]: {precision_score(Y_test, y_predict, average=parameter)*100}')


    for parameter in ['binary', None, 'micro', 'macro', 'weighted']:
        print(f'Recall [avg: {parameter}]: {recall_score(Y_test, y_predict, average=parameter)*100}')



# %% [markdown]
# ### Now we try our data on an Gaussian Naive Bayse

# %%
runCrossValOnModel(
            GaussianNB(), 
            [POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER]
)

runRandomSplitOnModel(
    GaussianNB(),
    [POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER]
)

# %% [markdown]
# #### Not good at all!

# %% [markdown]
# ### Let's try Multinomial one:

# %%
from sklearn.naive_bayes import MultinomialNB
runCrossValOnModel(
            MultinomialNB(alpha=0.9),
            [POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH, 
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)
runRandomSplitOnModel(
    MultinomialNB(),
    [POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER]
)

# %% [markdown]
# #### Not at all!

# %% [markdown]
# ### Let's try decision tree model

# %%
import sklearn.tree as tree
runCrossValOnModel(                                                                                                       # BEST
            tree.DecisionTreeClassifier(criterion='entropy'),           # add this parameter 96.99! time: 3.22
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,       
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

runRandomSplitOnModel(
    tree.DecisionTreeClassifier(criterion='entropy'),
    [IS_VERB, TOKEN, POS_TAG, NEXT_TOKEN_POS,
    PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,
    HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %% [markdown]
# # Good! 
# ### Now these parameters are the best found. you can check defferent parameters and their result below:

# %% [markdown]
# # In the end we picked the last 100000 for testing and the other for training. check there too!!!

# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,  # add TOKEN got better! 96.975 time: 2.59
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,  # remove "hasABiKolaah" a little worse 96.974 / time: 2.48
             HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,  # remove TOKEN_LENGTH got better! 96.985 time: 1.36
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER, IS_VERB,  # add IS_VERB a little worse! 96.984 / time: 1.40
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,  # remove hasVaav no difference! 96.985 time: 1.35
            HAS_AA_BIKOLAAH, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,  # remove hasHe much worse! 96.930 time: 1.32
            HAS_AA_BIKOLAAH, HAS_VAAV]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS,             # remove LAST_LETTER MUCH WORSE! 96.71 time: 2.24
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(),
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, LAST_LETTER,  # remove PREV_POS_TAG ogt worse! 96.969 time: 2.1
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)


# %%
runCrossValOnModel(                                                                                                       # BEST
            tree.DecisionTreeClassifier(criterion='entropy'),           # add this parameter 96.99! time: 3.22
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,       
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(criterion='entropy', max_depth=10),           # add max_dpth parameter got much worse!! 94!!!! time: 2.57
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,       
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runCrossValOnModel(
            tree.DecisionTreeClassifier(criterion='entropy', max_features="auto"),           # add max_feature='auto' 96.82! time:1.34
            [TOKEN, POS_TAG, NEXT_TOKEN_POS, PREV_TOKE_POS, LAST_LETTER,       
            HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runRandomSplitOnModel(
    tree.DecisionTreeClassifier(criterion='entropy'),
    [IS_VERB, TOKEN, POS_TAG, NEXT_TOKEN_POS,
    PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,
    HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runRandomSplitOnModel(
        tree.DecisionTreeClassifier(),
        [TOKEN, POS_TAG, NEXT_TOKEN_POS,            #removed IS_VERB  97.01 / 93.04 / 93.68
        PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,
        HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runRandomSplitOnModel(
        tree.DecisionTreeClassifier(),
        [TOKEN, POS_TAG, NEXT_TOKEN_POS,            #removed TOKEN_LENGTH: 97.00 / 93.16 / 93.56
        PREV_TOKE_POS, LAST_LETTER,
        HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE]
)

# %%
runRandomSplitOnModel(
        tree.DecisionTreeClassifier(),
        [TOKEN, POS_TAG, NEXT_TOKEN_POS,            #removed HAS_HE: 96.93 / 92.89 / 93.51
        PREV_TOKE_POS, LAST_LETTER,
        HAS_AA_BIKOLAAH, HAS_VAAV]
)

# %%
runRandomSplitOnModel(
        tree.DecisionTreeClassifier(),
        [TOKEN, POS_TAG, NEXT_TOKEN_POS,            #removed HAS_VAAV 97.006 / 93.07 / 93.66
        PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,
        HAS_AA_BIKOLAAH, HAS_HE]
)

# %% [markdown]
# ### At the end we define this function to take the last 100000 records for testing and the other for training

# %%
def runModelOnLastOneHundredThousand(model, Xs):
    X_train = df[Xs][ : -100000]
    X_test = df[Xs][-100000: ]
    Y_train = df[EZAFE_TAG][ : -100000]
    Y_test = df[EZAFE_TAG][-100000: ]

    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(Y_test, y_predict)}')

    for parameter in ['binary', None, 'micro', 'macro', 'weighted']:
        print(f'Precision [avg: {parameter}]: {precision_score(Y_test, y_predict, average=parameter)*100}')


    for parameter in ['binary', None, 'micro', 'macro', 'weighted']:
        print(f'Recall [avg: {parameter}]: {recall_score(Y_test, y_predict, average=parameter)*100}')

# %%
from sklearn.tree import DecisionTreeClassifier
runModelOnLastOneHundredThousand(DecisionTreeClassifier(criterion='entropy'),
    [IS_VERB, TOKEN, POS_TAG, NEXT_TOKEN_POS,
    PREV_TOKE_POS, LAST_LETTER, TOKEN_LENGTH,
    HAS_AA_BIKOLAAH, HAS_VAAV, HAS_HE])


