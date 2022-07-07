import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import pymorphy2 as pm

good = pd.read_csv('questions.csv')

#токенизируем и лемматизируем слова
def text_normalization(text):
  text=str(text).lower()
  spl_char_text=re.sub(r'[^a-яё0-9]',' ',text)
  tokens=nltk.word_tokenize(spl_char_text)
  lema_words=[]
  for i in tokens:
    i = pm.MorphAnalyzer().parse(i)[0]
    i=i.normal_form
    if i not in stopwords.words('russian'):
      lema_words.append(i)

  return ' '.join(lema_words)

good['lemmatized_text']=good['context_0'].apply(text_normalization)

#Векторизируем текст с помощью метрики tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(good.lemmatized_text)
matrix_big = vectorizer.transform(good.lemmatized_text)

#сокращение размерности
#импортируем алгоритм Метод главных компонент

from sklearn.decomposition import TruncatedSVD

#алгоритм будет проецировать данные в 300-мерное пространство

svd = TruncatedSVD(n_components=224)

svd.fit(matrix_big)
matrix_small =  svd.transform(matrix_big)

#поиск ближайщих соседей

import numpy as np
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator

#функция для создания вероятностного распределения
def softmax(x):
  proba = np.exp(-x)
  return proba/sum(proba)

#класс для случайного выбора одного из ближайших соседей

class NeighborSampler(BaseEstimator):
  def __init__(self, k=5, temperature=1.0):
    self.k = k
    self.temperature = temperature
  def fit(self, X, y):
    self.tree_ = BallTree(X)
    self.y_ = np.array(y)
  def predict(self, X, random_state=None):
    distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
    result = []
    for distance, index in zip(distances, indices):
      result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
      return self.y_[result]

from sklearn.pipeline import make_pipeline
ns = NeighborSampler()
ns.fit(matrix_small, good.reply)
pipe = make_pipeline(vectorizer, svd, ns)

"""##Публикация бота в телеграм##"""

import telebot
from secret import TOKEN

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])

def send_welcome(message):
  bot.reply_to(message, 'Я голос из окккупированного Яанислинна. Я помню все о военных годах в Карелии.\n' +
               'Оккупация Карелии войсками финской армии продлилась 1000 дней. Враг захватил почти половину территории республики. Треть населения отправили в концлагеря.\n' +
               '\n' +
               'Что ты хочешь узнать? Напиши свой вопрос. Постарайся сформулировать его как можно точнее, например, "оккупация Петрозаводска", "питание заключенных"' +
               '\n' +
               'Впрочем, задавай как хочешь.\n' +
               '\n' +
               'Если нужна справка, напиши /help' 
               )

@bot.message_handler(commands=['help'])  
def help_command(message):  
    keyboard = telebot.types.InlineKeyboardMarkup()  
    keyboard.add(  
        telebot.types.InlineKeyboardButton(  
            'Написать разработчикам', url='telegram.me/ag_utkina'  
  )  
    )  
    bot.send_message(  
        message.chat.id,  
        'Если ты хочешь что-то узнать об оккупации Петрозаводска в годы Великой Отечественной войны, просто напиши свой вопрос!).'
    )

@bot.message_handler(func=lambda message: True)

def echo_all (message):
  bot.reply_to(message, pipe.predict([text_normalization(message.text)])[0] +
               '\n' +  
               'Чат-бот создан в рамках проекта "Яанислинна: город несвободы". Если нужна помощь, напиши /help'
               )

bot.polling()