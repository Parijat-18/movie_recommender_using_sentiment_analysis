from bs4 import BeautifulSoup
import numpy as np
from urllib.parse import urljoin
import requests

import numpy as np
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def get_imdb_reviews(url, count=5, sorted=True,tokenized=False):
    response = requests.get(url , headers=headers)
    assert response.status_code == 200

    html_file = response.text
    soup = BeautifulSoup(html_file, 'lxml')

    all_reviews_html = soup.find('div', class_='lister-list')
    if all_reviews_html is None:
        print('No Reviews Found')
        return

    all_reviews = []
    for review_html in all_reviews_html.find_all('div', class_='lister-item mode-detail imdb-user-review collapsable'):
      
      if count:
        review = review_html.find('div', class_='text show-more__control')
        if review is not None:
          all_reviews.append(review.text)
          count -= 1

    return np.asarray(all_reviews)



def get_genres(movie_url):
  response = requests.get(movie_url , headers=headers)
  genres = []

  assert response.status_code == 200

  movie_html_file = response.text
  soup = BeautifulSoup(movie_html_file , 'lxml')
  try:
    genres_html = soup.find('div' , class_='ipc-chip-list__scroller')
    for genre in genres_html.find_all('span' , class_='ipc-chip__text'):
      genres.append(genre.text)
    return genres
  except:
    return []
  

  

def get_imdb_mov_rev_link( movie_name):
    base_url = "https://www.imdb.com/"
    
    def get_html_text(url):
        response = requests.get(url , headers=headers)
        if response.status_code == 200:
            return BeautifulSoup(response.text , 'lxml')
        else:
            print('URL Not Found')
            return None

    search_url = urljoin(base_url , f'/find/?q={"%20".join(movie_name.split(" "))}')
    search_html_text = get_html_text(search_url)

    if search_html_text is not None:
        movie_url = search_html_text.find('a' , class_='ipc-metadata-list-summary-item__t')
        if movie_url is not None:
            movie_url =urljoin(base_url, movie_url.get('href'))
            movie_html_text = get_html_text(movie_url)
            reviews_header = movie_html_text.find('div', {'data-testid': 'reviews-header'})
            if reviews_header is not None:
                review_url = urljoin(base_url, movie_url)[:-16] + 'reviews?sort=totalVotes'
                return movie_url, review_url
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None