from secrets import song, song_times, artist, streams
import requests as rq
from bs4 import BeautifulSoup
import lxml
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction import text
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob as tb
import math
from gensim import matutils, models
import scipy.sparse
import time


# 1.0 - Gathering Data
def lyric_retrieve(url):
    page = rq.get(url).text
    page = page.replace("<br>", " ")
    page = page.replace("<br/>", " ")
    page = page.replace("\u2005", " ")
    soup = BeautifulSoup(page, "lxml")
    lyrics = [div.text for div in soup.find_all("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")]
    if not lyrics:
        lyrics.append("matcha")  # matcha returns 0 for sentiment index, placeholder for inability to scrape lyrics
    return lyrics


# 1.1 - Cleaning Data
def stringify(lyrics):
    string_text = " ".join(lyrics)
    return string_text


def df_time(dict):
    songs_df = pd.DataFrame.from_dict(dict).transpose()
    songs_df.columns = ["lyrics"]
    return songs_df


def data_clean_one(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub("ooh", "", text)
    return text


def corpus_word_remove(corpus):
    stopwords = ["ooh"]
    corpus_new = [word for word in corpus.lyrics if word == stopwords]

    return corpus_new


def stop_word_create():
    wanted_words = ["you", "cry"]
    extra_words = ["ooh", "the", "a", "i", "and", "of", "oh", "yeah", "you"]
    all_words = text.ENGLISH_STOP_WORDS.union(extra_words)
    stop_words = [words for words in all_words if words not in wanted_words]

    return stop_words


def dtm(clean, stopWords):
    words = cv(stop_words=stopWords)
    data_cv = words.fit_transform(clean.lyrics)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=words.get_feature_names())
    data_dtm.index = clean.index
    return data_dtm


# 2.0 - Exploratory Data Analysis
# 2.1 - Most Common Words
def top_words(dtm):
    ordered = dtm.transpose()
    topWords = {}
    for c in ordered.columns:
        top = ordered[c].sort_values(ascending=False).head(10)
        topWords[c] = list(zip(top.index, top.values))
    return topWords


def words_per_5s(data, artist):
    word_count = []
    song_time = song_times
    song_time_5s = [i * 12 for i in song_time]  # number of 5s intervals in the song
    word_per_5s = []

    for i in range(len(artist)):
        word_count.append(data.values[i].sum())
        word_per_5s.append(word_count[i] / song_time_5s[i])
        if word_per_5s[i] < 0.4:
            word_per_5s[i] = 0

    return word_per_5s


# 3.0 - Sentiment Analysis
# 3.1 - flat sentiment analysis
def sent_analysis(corpus):
    pol = lambda x: tb(x).sentiment.polarity
    corpus["polarity"] = corpus["lyrics"].apply(pol)
    return corpus


# 3.2 - sentiment analysis over song
def split_data(corpus, n=3):
    length = len(corpus)
    size = math.floor(length / n)
    start = np.arange(0, length, size)

    split_list = []
    for piece in range(n):
        split_list.append(corpus[start[piece]:start[piece]+size])
    return split_list


def senti_analysis_time(split_lyrics):
    polarity_transcript = []
    for piece in split_lyrics:
        polarity_piece = []
        for lyrics in piece:
            polarity_piece.append(tb(lyrics).sentiment.polarity)
        polarity_transcript.append(polarity_piece)
    return polarity_transcript


if __name__ == '__main__':
    start_time = time.time()

    urls = song
    songs = [lyric_retrieve(u) for u in urls]
    artists = artist

    pd.set_option("max_colwidth", 100)
    data = {}
    for i, c in enumerate(artists):
        data[c] = songs[i]

    data_combine = {key: [stringify(value)] for (key, value) in data.items()}

    songs_df = df_time(data_combine)
    data_clean1 = pd.DataFrame(songs_df.lyrics.apply(lambda x: data_clean_one(x)))

    stop_list = stop_word_create()
    data_dtm = dtm(data_clean1, stop_list)  # DTM

    topWords = top_words(data_dtm)  # List of Top Words

    # Number of words every 5 seconds
    word_rate = words_per_5s(data_dtm, artists)
    word_rate_df = pd.DataFrame(list(zip(artists,word_rate)), columns=["artist", "word_rate (per 5s)"])
    print("- - - Words Per 5 Seconds - - -\n", word_rate_df)

    # Sentiment analysis for polarity
    data_corpus_senti = sent_analysis(data_clean1)

    print("\n- - - Sentiment Index - - -")
    for i in range(len(artists)):
        print(artists[i], end="")
        print(": ", data_corpus_senti["polarity"][i])  # Sentiment Index

    # Sentiment analysis over time
    split_lyrics = []
    for t in data_clean1.lyrics:
        split_lyrics.append(split_data(t))

    sentiment_rate = senti_analysis_time(split_lyrics)
    senti_rate_p = [y for x in sentiment_rate for y in x]
    # print("The Sentiment Rate for each Song: ", senti_rate_p)  # Sentiment Index Rate


    print("\n- - - Runtime - - -\n- %s seconds -" % (time.time() - start_time))  # Total Runtime

    # Creating word clouds
    data_clean1_string = [str(lyrics) for lyrics in data_clean1.lyrics]
    clouds = []
    for i in range(len(artist)):
        clouds.append(wc(stopwords = stop_list, background_color = "white", colormap = "Dark2",
                         max_font_size=150, random_state = 42).generate(data_clean1_string[i]))
        plt.imshow(clouds[i], interpolation="bilinear")
        plt.axis("off")
        plt.title(artist[i], fontsize=20)
        plt.show()

#TOPIC MODELING
    sparse_counts = scipy.sparse.csr_matrix(data_dtm)
    corpus = matutils.Sparse2Corpus(sparse_counts)

    stopWords = stop_word_create()

    words = cv(stop_words=stopWords)
    data_cv = words.fit_transform(data_clean1.lyrics)

    id2word = dict((v, k) for k, v in words.vocabulary_.items())
    lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
    # print(lda.print_topics())

    x_senti = np.array(data_corpus_senti["polarity"])
    y_streams = np.array(streams)

    plt.scatter(x_senti, y_streams, marker=True)
    plt.title("sentiment index vs popularity", fontsize=14)
    for i, txt in enumerate(artist):
        plt.annotate(txt, (x_senti[i], y_streams[i]))
    plt.show()











