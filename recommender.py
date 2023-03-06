import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from imdb_scraper import headers , get_imdb_reviews , get_imdb_mov_rev_link , get_genres


GENRES = ['Action', 'Adventure', 'Animation', 'Biography','Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Superhero', 'Thriller', 'War', 'Western']


def preprocess(df, output_numpy_path, model, tokenized= False, tokenizer= None, return_tensors='pt', count= 5):

    def preprocess_df(row):
        movie_url , review_url = get_imdb_mov_rev_link(row['title'])
        if review_url != None:
            if tokenized == True:
                sentiment = reviews_to_sent(review_url , model , tokenized=True , count=count)
            else:
                sentiment = reviews_to_sent(review_url , model , tokenized=False , tokenizer=tokenizer , return_tensors=return_tensors, count=count)
            if sentiment is not None:
                row['sentiment'] = sentiment
            else:
                return False

        if row['genres'] != "(no genres listed)":
            genresdf = row['genres'].split('|')
            for genre in genresdf:
                row[genre] = 1
        else:
          return False
        pbar.update(1)
        return row

    pbar = tqdm(total=len(df))
    df = df.apply(preprocess_df , axis=1) 
    df.fillna(0, inplace=True)

    # Save preprocessed data as numpy file
    np.save(output_numpy_path, df.to_numpy())
    
    return df


def reviews_to_sent(rev_url , model , tokenized=True , tokenizer=None, return_tensors='tf', count=5):
  response = requests.get(rev_url , headers=headers)
  assert response.status_code == 200
  if tokenized == False: assert tokenizer != None

  reviews = get_imdb_reviews(rev_url , count=count)
  if reviews is not None and (reviews.size != 0):
    if tokenized == False:
      reviews = reviews.tolist()
      encoded_rev = tokenizer(reviews, padding = True, truncation=True, return_tensors=return_tensors)
      if return_tensors == 'tf':
        predictions = model(**encoded_rev).logits.numpy()
      else:
        predictions = model(**encoded_rev).logits.detach().numpy() 
    else:
      predictions = model.predict(reviews , verbose=0)
    predictions = np.mean(predictions , axis=0)
    return predictions
  else:
    return None


from sklearn.metrics.pairwise import cosine_similarity

def get_mov_ranked(preprocessed_df , movie_name , model , tokenizer=None , return_tensors="tf" , mov_count=10):
  movies = preprocessed_df['title'].to_numpy()
  all_genres = preprocessed_df[GENRES].to_numpy()
  sentiments = preprocessed_df['sentiment'].apply(pd.Series).to_numpy()

  mov_link , rev_link = get_imdb_mov_rev_link(movie_name)
  mov_genres = get_genres(mov_link)
  if tokenizer == None:
    mov_sentiment = reviews_to_sent(rev_link , model , tokenized=True , count=5).reshape(1 , -1)
  else:
    mov_sentiment = reviews_to_sent(rev_link , model , tokenized=False , tokenizer=tokenizer , return_tensors=return_tensors , count=5).reshape(1 , -1)

  sparsed_mov_genres = np.asarray([int(genre in mov_genres) for genre in GENRES]).reshape(1 , -1)
  ranked_idxs = np.argsort(cosine_similarity(all_genres , sparsed_mov_genres).flatten())[::-1]

  movies_ranked = movies[ranked_idxs][:mov_count]
  sentiments_ranked = sentiments[ranked_idxs][:mov_count]
  
  sent_ranked_idxs = np.argsort(cosine_similarity(sentiments_ranked , mov_sentiment).flatten())[::-1]
  
  return movies_ranked[sent_ranked_idxs]



