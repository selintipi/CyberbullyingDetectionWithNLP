import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import re
import unidecode
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import sklearn
from sklearn.ensemble import RandomForestClassifier

# PREPROCESS FUNCTIONS

# tekrarlayanları sil
def delete_duplicates(data):
    print("\nThe number of duplicates: %d" % data.duplicated().sum())
    data = data.drop_duplicates()
    print("The number of duplicates (AFTER DELETING): %d" % data.duplicated().sum())
    return data


# tüm herfleri küçük harf yap
def convert_to_lowercase(data):
    data.tweet_text = [i.lower() for i in data.tweet_text.values]
    return data


# @,! gibi özel karakterleri kaldır
def remove_specials(data):
    data.tweet_text = [
        re.sub(r"[^a-zA-Z]", " ", text) for text in data.tweet_text.values
    ]
    return data


# ingilizce'de kısa hali kullanılanları uzunları ile değiştir
def remove_shorthands(data):
    CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }
    texts = []
    for text in data.tweet_text.values:
        string = ""
        for word in text.split(" "):
            if word.strip() in list(CONTRACTION_MAP.keys()):
                string = string + " " + CONTRACTION_MAP[word]
            else:
                string = string + " " + word
        texts.append(string.strip())
    data.tweet_text = texts
    return data


# a, the, this gibi kelimeleri kaldır
def remove_stopwords(data):
    texts = []
    stopwords_list = stopwords.words("english")
    for item in data.tweet_text.values:
        string = ""
        for word in item.split(" "):
            if word.strip() in stopwords_list:
                continue
            else:
                string = string + " " + word
        texts.append(string)
    data.tweet_text = texts
    return data


# web adreslerini kaldır
def remove_links(data):
    texts = []
    for text in data.tweet_text.values:
        remove_https = re.sub(r"http\S+", "", text)
        remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
        texts.append(remove_com)
    data.tweet_text = texts
    return data


# yumuşak a gibi aksan harflerini normalleriyle değiştir
def remove_accents(data):
    data.tweet_text = [unidecode.unidecode(text) for text in data.tweet_text.values]
    return data


# birden fazla boşluk kullanıldıysa düzelt
def normalize_spaces(data):
    data.tweet_text = [re.sub(r"\s+", " ", text) for text in data.tweet_text.values]
    return data


# MAIN CODE

# csv formatındaki veri setini oku
data = pd.read_csv("cyberbullying_tweets.csv")

print("\nDataset Info:")
data.info()

# Print raw data
print("\nRaw Data:")
print(data)

# preprocessleri çalıştır
data = delete_duplicates(data)
data = convert_to_lowercase(data)
data = remove_links(data)
data = remove_shorthands(data)
data = remove_accents(data)
data = remove_specials(data)
data = remove_stopwords(data)
data = normalize_spaces(data)

# Print processed data
print("\nProcessed Data:")
print(data)

# Draw a pie chart for data types
data.groupby("cyberbullying_type").size().plot(kind="pie")
plt.show()

# fit transform, normalizes data
lenc = sklearn.preprocessing.LabelEncoder()
data.cyberbullying_type = lenc.fit_transform(data.cyberbullying_type)

print()
for c in range(len(lenc.classes_)):
    print(c, lenc.classes_[c])
    string = ""
    # tüm ilgili kelimelerin olduğu bir string oluşturur
    for i in data[data.cyberbullying_type == c].tweet_text.values:
        string = string + " " + i.strip()

    # Generate the wordcloud
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=set(STOPWORDS),
        min_font_size=10,
    ).generate(string)

    # Plot the wordcloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(lenc.classes_[c])
    plt.show()
    del string

# Turn raw data into TF-IDF matrix
vec = sklearn.feature_extraction.text.TfidfVectorizer(max_features=3000)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    vec.fit_transform(data.tweet_text.values).toarray(),
    data.cyberbullying_type.values.reshape(-1, 1),
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

rand_forest_model = RandomForestClassifier(n_estimators=10, random_state=42)
rand_forest_model.fit(X_train, Y_train.ravel())

log_reg_model = sklearn.linear_model.LogisticRegression(max_iter=200)
log_reg_model.fit(X_train, Y_train.ravel())

print("\nLogistic Regression Final Results:")
print(
    "Train Accuracy  : {:.2f} %".format(
        sklearn.metrics.accuracy_score(log_reg_model.predict(X_train), Y_train) * 100
    )
)
print(
    "Test Accuracy   : {:.2f} %".format(
        sklearn.metrics.accuracy_score(log_reg_model.predict(X_test), Y_test) * 100
    )
)
print(
    "Precision       : {:.2f} %".format(
        sklearn.metrics.precision_score(
            log_reg_model.predict(X_test), Y_test, average="macro"
        )
        * 100
    )
)
print(
    "Recall          : {:.2f} %".format(
        sklearn.metrics.recall_score(
            log_reg_model.predict(X_test), Y_test, average="macro"
        )
        * 100
    )
)
print("\nRandom Forest Final Results:")
print(
    "Train Accuracy  : {:.2f} %".format(
        sklearn.metrics.accuracy_score(rand_forest_model.predict(X_train), Y_train)
        * 100
    )
)
print(
    "Test Accuracy   : {:.2f} %".format(
        sklearn.metrics.accuracy_score(rand_forest_model.predict(X_test), Y_test) * 100
    )
)
print(
    "Precision       : {:.2f} %".format(
        sklearn.metrics.precision_score(
            rand_forest_model.predict(X_test), Y_test, average="macro"
        )
        * 100
    )
)
print(
    "Recall          : {:.2f} %".format(
        sklearn.metrics.recall_score(
            rand_forest_model.predict(X_test), Y_test, average="macro"
        )
        * 100
    )
)

cm = sklearn.metrics.confusion_matrix(Y_test, rand_forest_model.predict(X_test))
disp = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=lenc.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.show()
