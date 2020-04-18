from app import re
from app import tf
from app import hub
from app import cosine_similarity
from app import pd
from app import np

def PreprocessInput(area, keyword): 
    area = area.lower()
    area = area.replace(r"[^A-Za-z0-9^,!.\/'+-=]", " ")
    area = area.replace(r"what's", "what is ")
    area = area.replace(r"\'s", " ")
    area = area.replace(r"\'ve", " have ")
    area = area.replace(r"can't", "cannot ")
    area = area.replace(r"n't", " not ")
    area = area.replace(r"i'm", "i am ")
    area = area.replace(r"\'re", " are ")
    area = area.replace(r"\'d", " would ")
    area = area.replace(r"\'ll", " will ")
    area = area.replace(r",", " ")
    area = area.replace(r"\.", " ")
    area = area.replace(r"!", " ! ")
    area = area.replace(r"\/", " ")
    area = area.replace(r"\^", " ^ ")
    area = area.replace(r"\+", " + ")
    area = area.replace(r"\-", " - ")
    area = area.replace(r"\=", " = ")
    area = area.replace(r"'", " ")
    area = area.replace(r"(\d+)(k)", r"\g<1>000")
    area = area.replace(r":", " : ")
    area = area.replace(r" e g ", " eg ")
    area = area.replace(r" b g ", " bg ")
    area = area.replace(r" u s ", " american ")
    area = area.replace(r"\0s", "0")
    area = area.replace(r" 9 11 ", "911")
    area = area.replace(r"e - mail", "email")
    area = area.replace(r"j k", "jk")
    area = area.replace(r"\s{2,}", " ")
    area = [area]
    print(area)
    if (keyword != ''):
        keyword = keyword.split(',')

    return area, keyword


def ElmoInput(area):
    m = hub.KerasLayer('/home/Mathieu23IA/mysite/app/data/1')
    area_vect = m(tf.convert_to_tensor(area))
    """embed = hub.Module('app/data/1', trainable=True)
    embedding = embed(area)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        area_vect = sess.run(embedding)"""

    return area_vect


def CosSimi(nb_result, area_vect, data_vect):
    similarite = pd.Series(cosine_similarity(area_vect, data_vect).flatten())
    """ list_n_simi is an array if the higher value at the end  (ie: the len(database)-1 value)"""
    return similarite

def SelectKeyword(data_csv, keyword):
    list_blurb = data_csv['Blurb'].tolist()
    occurence = 0
    max_occ = 0
    for i in range(0, len(keyword)):
        occurence = list_blurb.count(keyword[i])
        if (occurence >= max_occ):
            max_occ = occurence
            indice = i
    
    if (occurence >= 3):
        final_word = keyword[indice]
    else:
        final_word = ''

    return final_word

def FinalResult(final_word, similarite, data_csv, data_csv_display, size_data, nb_result):
    five_blurb = []
    five_title = []
    five_isbn = []
    five_author = []
    score_display =[]
    if (final_word != ''):
        nb_display = 0
        for i in range(0,nb_result):
            best_index = np.argmax(similarite)
            score_display.append(similarite[best_index])
            similarite[best_index]=0
            if(nb_display <= 5):
                if (final_word in data_csv['Blurb'][best_index]):
                    nb_display = nb_display + 1
                    five_blurb.append(data_csv_display['Blurb'][best_index])
                    five_title.append(data_csv_display['Title'][best_index])
                    five_isbn.append(data_csv_display['ISBN'][best_index])
                    five_author.append(data_csv_display['Author'][best_index])

    else:
        for i in range(0,5):
            best_index = np.argmax(similarite)
            score_display.append(similarite[best_index])
            similarite[best_index]=0
            five_blurb.append(data_csv_display['Blurb'][best_index])
            five_title.append(data_csv_display['Title'][best_index])
            five_isbn.append(data_csv_display['ISBN'][best_index])
            five_author.append(data_csv_display['Author'][best_index])

    return five_blurb, five_title, five_isbn, five_author, score_display