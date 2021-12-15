import pandas as pd
import numpy as np
import tweepy
import matplotlib.pyplot as plt

# Sastrawi Package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning package
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import string
import re
from string import punctuation
from collections import Counter

# SNA
import networkx as nx
from networkx.readwrite import json_graph
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

#main function
def main(request):
  keywords = request.form["keywords"]
  tnum =request.form["tnum"]
  # Call twitter api keys
  consumer_key = 'xxxxxxxxxxxxxx'
  consumer_secret = 'xxxxxxxxxxx'
  access_token = 'xxxxxxxxxxxxxxx'
  access_secret = 'xxxxxxxxxxxxxxx'
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  api = tweepy.API(auth)
  # Search keywords
  tweets = tweepy.Cursor(api.search,q=keywords,tweet_mode="extended",lang="id").items(int(tnum))
  message,retweet_count,retweet,created_at,user_name,user_id=[],[],[],[],[],[]
  count = 0
  for tweet in tweets:
      count=count+1
      if hasattr(tweet, 'retweeted_status'):
          message.append(tweet.retweeted_status.full_text)
          retweet_count.append(tweet.retweet_count)
          retweet.append(tweet.retweeted_status.user.screen_name)
          created_at.append(tweet.created_at)
          user_name.append(tweet.user.screen_name)
          user_id.append(tweet.user.id)
      else:
          message.append(tweet.full_text)
          retweet_count.append(tweet.retweet_count)
          retweet.append(print(''))
          created_at.append(tweet.created_at)
          user_name.append(tweet.user.screen_name)
          user_id.append(tweet.user.id)
  # insert tweets to database
  for i in range(count):
      data=[message[i], retweet_count[i], retweet[i], created_at[i], user_name[i], user_id[i]]
  # make dataframe
  df=pd.DataFrame({
      'author':retweet,
      'username':user_name,
      'retweet_count':retweet_count,
      'tweets':message,
      'created_at':created_at
  })
  df = df.sort_values(['created_at'], ascending=[0])
  df1 = df.copy()
  # helper function to clean tweets
  def preprocessing(tweet):
      tweet=re.sub('<.*?>','',tweet)
      tweet = re.sub('@[^\s]+','',tweet)
      tweet = re.sub(r'\$\w*', '', tweet)
      tweet=tweet.lower()
      tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
      tweet=re.sub(r'#[A-Za-z0-9_]+','',tweet)
      tweet=re.sub(r'[^\w\s]','',tweet)
      tweet=re.sub(r'\b\w{1,2}\b', '', tweet)
      tweet=re.sub(r'\s\s+', ' ', tweet)
      tweet = re.sub(r'\s\s+', ' ', tweet)
      tweet=''.join(c for c in tweet if c <= '\uFFFF')
      return tweet
    
  # clean dataframe's text column
  df['tweets'] = df['tweets'].apply(preprocessing)
    
  # drop duplicates
  df = df.drop_duplicates('tweets')
  factory = StopWordRemoverFactory()
  stopword = factory.create_stop_word_remover()
  
  def stoptweet(tweet):
    tweet = stopword.remove(tweet)
    replace_list = ['wow semua']
    tweet = re.sub(r'|'.join(map(re.escape, replace_list)), '', tweet)
    return tweet

  df['tweets'] = df['tweets'].apply(stoptweet)
    
  # create stemmer
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  def stemtweet(tweet):
    tweet = stemmer.stem(tweet)
    return tweet

  # clean dataframe's text column
  df['tweets'] = df['tweets'].apply(stemtweet)
  lst = ['']
  lst = [x.strip() for x in lst] 

  # tokenize helper function
  def tokenization(tweet):
    tweet=tweet.split(' ')
    return tweet

  df['tokens'] = df['tweets'].apply(tokenization) # tokenize style 1
  df = df[['tweets','tokens']]
  all_words = []
  for line in df['tokens']: 
      all_words.extend(line) 
        
  # wordcloud step
  wordfreq = Counter(all_words)
  wordcloud = WordCloud(width=900,
                        height=500,
                        max_words=500,
                        max_font_size=100,
                        relative_scaling=0.5,
                        colormap='gist_rainbow',
                        normalize_plurals=True).generate_from_frequencies(wordfreq)
  plt.figure(figsize=(17,14))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.savefig('static/assets/img/wc.png')
  model = joblib.load("model_sentiment.pkl")

  # append predictions to dataframe
  prediction=model.predict(df['tweets'])
  result={'tweets':df['tweets'],'sentiment':prediction}
  result=pd.DataFrame(result)
  pos = result.sentiment.value_counts()[0]
  neg = result.sentiment.value_counts()[1]

  print('Model predictions: Positives - {}, Negatives - {}'.format(neg,pos))

  import plotly.graph_objs as go

  labels = ['Positif','Negatif']
  values = [int(pos),int(neg)]
  posneg = {'data' : [{'type' : 'pie', 
                       'name' : "Students by level of study",  
                       'labels' : labels,
                       'values' : values,
                       'direction' : 'clockwise',
                       'marker' : {'colors' : ["rgb(251,57,88)", "rgb(0,64,255)"]}}],
                      'layout' : {'title' : ''}}
  sa = plot(posneg,config={"displayModeBar": False}, 
                 show_link=False, 
                 include_plotlyjs=False, 
                 output_type='div')
  # Remove null account
  df1=df1.dropna()
  # Netwrokx
  net = nx.from_pandas_edgelist(df1, source="author", target="username")
  # Plot it
  G = nx.convert_node_labels_to_integers(net, first_label=0, ordering='default', label_attribute=None)
  pos=nx.fruchterman_reingold_layout(G)
  #create labels
  poslabs=nx.fruchterman_reingold_layout(net)
  labels=list(poslabs) + list(' : ')
  #create edges
  Xe=[]
  Ye=[]
  for e in G.edges():
      Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
      Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
  trace_edges=dict(type='scatter',
                  mode='lines',
                  x=Xe,
                  y=Ye,
                  line=dict(width=1, color='rgb(25,25,25)'),
                  hoverinfo='none' 
                  )

  #create nodes
  Xn=[pos[k][0] for k in range(len(pos))]
  Yn=[pos[k][1] for k in range(len(pos))]
  trace_nodes=dict(type='scatter',
                  x=Xn, 
                  y=Yn,
                  mode='markers',
                  marker=dict(showscale=True,size=5,color=[],colorscale='Rainbow',reversescale=True,colorbar=dict(
                      thickness=15,
                      title='Node Connections',
                      xanchor='left',
                      titleside='right')),
                  text=labels,
                  hoverinfo='text')

  #scale color by size
  for node, adjacencies in enumerate(G.adjacency()):
      trace_nodes['marker']['color']+=tuple([len(adjacencies[1])])
  #plot
  axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='' 
            )
  layout=dict(title= 'Social Network Analysis',  
              font= dict(family='Balto'),
              width=1000,
              height=1000,
              autosize=False,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40,r=40,b=85,t=100,pad=0,
              ),
              hovermode='closest',
  #     plot_bgcolor='#000000',           
      )
  fig = dict(data=[trace_edges,trace_nodes], layout=layout)
  #run plot
  sna = plot(fig,config={"displayModeBar": False},show_link=False, include_plotlyjs=False, output_type='div')
  snatable = ff.create_table(df1)
  snatab = plot(snatable,config={"displayModeBar": False}, 
                show_link=False, 
                include_plotlyjs=False, 
                output_type='div')
          # Save plot to html
  html_string = '''
        <html>
            <head>
              <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
              <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.0/css/bootstrap.min.css">
            </head>
            <body>
              <div>
                  <div class = "container">
                      <div class = "row">
                          <div class="col-md-6">
                             <h2>Social Network Analysis </h2> 
                             ''' + sna +'''
                          </div>
                          <div class="col-md-6">
                             <h2>Sentiment Analysis </h2> 
                             ''' + sa +'''
                          </div>
                          <div class="col-md-12">
                              <h2>WordCloud </h2> 
                              <img src='/static/assets/img/wc.png' width="900" height="600">
                          </div>
                          <div class="col-md-12">
                          ''' + snatab +'''
                          </div>
                      </div>
                  </div>
              </div>
            </body>
        </html>'''
        

  with open("templates/out.html", 'w') as f:
      f.write(html_string)
