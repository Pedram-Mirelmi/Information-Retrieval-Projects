# run with python 3.9 or higher to avoid error with type hints or remove them manualy
# a class containing each posting that stores a token with its frequency
class Posting:
    def __init__(self, token, frequency) -> None:
        self.token = token
        self.frequency = frequency

# now we define some operator overloading to make the way for python's list.sort() to sort a list of "Postings" 

    def __lt__ (self, other) -> bool:
        return self.frequency < other.frequency
    
    def __gt__ (self, other) -> bool :
        return self.frequency > other.frequency
    
    def __le__ (self, other) -> bool:
        return self.frequency <= other.frequency
    
    def __ge__ (self, other) -> bool:
        return self.frequency >= other.frequency
    
    def __eq__ (self, other) -> bool:
        return self.frequency == other.frequency
    
    def __ne__ (self, other) -> bool:
        return self.frequency != other.frequency
    
    def __repr__(self) -> str:
        return f"\"{self.token}\": {self.frequency}"
    
# and overload the float() and int() functions so plt.plot() can accept the X as a list of "Postings"

    def __float__(self) -> float:
        return float(self.frequency)

    def __int__(self) -> int:
        return int(self.frequency)

# a generator to return a token whenever we need. later we iterate over it
def nextToken() -> str:  # gerenator 
    global text
    i = 0
    reading_valid_token = False
    for j, chr in enumerate(text):
        if chr.isalpha():
            reading_valid_token = True
        elif reading_valid_token:
            yield text[i:j].lower()
            i = j + 1
            reading_valid_token = False
        else:
            i = j + 1
    return ''


# just a simple perfomance messure
import time
beggin_time = time.perf_counter()

# open the text file
with open("./shakespear/text.txt") as dataset_file:
    text = dataset_file.read()


# a list containing the number of total words in each step. it will be always like this: [1, 2, 3, 4, 5, 6, 7]
all_words_count = [1]

# a list contating the number of unique words in each step. for example when indexing "Hi Hello Hi", at the end it will be: [1, 2, 2]
new_words_count = [1]

# a dict contaning words read so far and their frequency
known_words = {nextToken(): 1}


import math
# indexing for loop
for token in nextToken():
    if token in known_words: # if current token has been read so far(not a new word)
        known_words[token] += 1 # increment its frequency
        all_words_count.append(all_words_count[-1] + 1) # all_words will always be like: [1, 2, 3, 4, 5, 6, 7, ...]
        new_words_count.append(new_words_count[-1]) # number of unique words won't be incremented
    else:
        known_words[token] = 1  # add the new word to dictionary
        all_words_count.append(all_words_count[-1] + 1) # all_words will always be like: [1, 2, 3, 4, 5, 6, 7, ...]
        new_words_count.append(new_words_count[-1] + 1) # increment the number of unique words


# print some info and perfomance messure result
print(f"read file and indexed {len(all_words_count)} words in {time.perf_counter() - beggin_time} seconds!")


import matplotlib.pyplot as plt
import math

# Heap law

# prepare X and Y data cordinates to plot the Heap law
X_cordinate = [math.log10(i) for i in all_words_count]
Y_cordinate = [math.log10(i) for i in new_words_count]
plt.plot(X_cordinate, Y_cordinate)  # ploted here
plt.title("Heap law")               # set the title
plt.ylabel("New(unique) words")     # y will show growth of number of unique words in progress
plt.xlabel("Total words")           # x will show the number of total words
from scipy import stats
slope, intercept, r, p, std_err = stats.linregress(X_cordinate, Y_cordinate) # estimate a line througth the data
print(f"line equation: Y = {slope} X + {intercept}")                         # printing line equation
plt.plot(X_cordinate, [slope*x + intercept for x in X_cordinate])            # ploting the line
plt.legend(["Original data", "Predicted"])                                   # identify each graph
plt.show()                                                                   # show the plot. when window closes the rest of code will be executed

# create a list of "Posting" objects using knows_words dictionary
sorted_postings = [Posting(*posting) for posting in known_words.items()]
sorted_postings.sort(reverse=True)  # sort it. (by the operator overloading we did it will be sorted by frequency)

# print some info about the data
print("Top 20 most frequent tokens:")
print(*sorted_postings[:20], sep='\n')



# zipf law:
# ploting and finding best line using linear regression just the same as Heap law

X_cordinate = [math.log10(i) for i in range(1, len(sorted_postings)+1)]
Y_cordinate = [math.log10(posting) for posting in sorted_postings]
plt.plot(X_cordinate, Y_cordinate)
slope, intercept, r, p, std_err = stats.linregress(X_cordinate, Y_cordinate)
print(f"line equation: Y = {slope} X + {intercept}")
plt.plot(X_cordinate, [slope*x + intercept for x in X_cordinate])
plt.legend(["Original data", "Predicted"])
plt.show()
