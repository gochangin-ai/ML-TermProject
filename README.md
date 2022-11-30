# ML-TermProject

## recommendation system
+ Collaborative filtering
+ Content-based filtering
## [Dataset](https://github.com/gochangin-ai/ML-TermProject/blob/main/README.md#Dataset) <
## [Code with User Manual](https://github.com/gochangin-ai/ML-TermProject/blob/main/README.md#code-with-user-manual) <
## [Results](https://github.com/gochangin-ai/ML-TermProject/blob/main/README.md#results) <



# Dataset
## MovieLens 20M Dataset
#### Context
The datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on October 17, 2016.

Users were selected at random for inclusion. All selected users had rated at least 20 movies.

※you have to download rating.csv and put in 'input' folder to use this recommendation system ! ※


# Code With User Manual
## Importing necessary libraries
#### Make sure that you have all these libaries available to run the code successfully
```python
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from  sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from fuzzywuzzy import process

warnings.filterwarnings(action='ignore')

movies = pd.read_csv('input/movies_train.csv')
ratings = pd.read_csv('input/ratings_train.csv')
movies1 = movies.copy()
pd.set_option('display.max_columns', None)
```


## Content-Based Filtering using KNN
reference:
1. https://www.kaggle.com/code/yogeshrampariya/content-based-recommendation-system

### Function : content_based_preprocessing(mov)
#### Delete the ‘genres’ column of the existing movie dataset and create a new column for all genres. And if each movie belongs to the corresponding genre column, the value is displayed as 1, otherwise it is displayed as 0.
##### Parameter:
##### mov | dataframe
##### Return:
##### gen_mat | dataframe
```python
def content_based_preprocessing(mov):
    genre = mov['genres'].str.split('|', expand=True)
    print(genre)
    gen_mat = mov[['movieId', 'genres']].copy()
    gen_mat['genres'] = gen_mat['genres'].str.lower().str.strip()
    print(gen_mat.head(5))

    all_gen = List_All_Genres(movies1)
    print(all_gen)

    for gen in all_gen:
        gen_mat[gen] = np.where(gen_mat['genres'].str.contains(gen), 1, 0)

    pd.set_option('display.max_columns', None)
    print(gen_mat.head(3))

    # Drop genres
    gen_mat.drop('genres', axis=1, inplace=True)
    gen_mat = gen_mat.set_index('movieId')
    print(gen_mat.head(3))

    return gen_mat
```

### Function : List_All_Genres(movies)
#### Extract all genres of movies and save them to a list.
##### Parameter:
##### movies | dataframe
##### Return:
##### genres | list
```python
def List_All_Genres(movies):
    genres = []
    for genre in movies.genres:
        x = genre.split('|')
        for i in x:
            if (str(i).lower() not in genres):
                my_genres = str(i).lower()
                genres.append(my_genres)
    return genres
```

### Function : content_based_model_build(gen_, my_list, top_k, map_name)
#### Calculate the genre table obtained from content_based_preprocessing(mov) using cosine similarity. And recommend movies by selecting k with the highest similarity.
##### Parameter:
##### gen | dataframe
##### my_list | list
##### top_k | int
##### map_name | dict
```python
def content_based_model_build(gen_, my_list, top_k, map_name):
    for item_id in my_list:
        print('Movie Selected : {0}'.format(movies1.values[item_id]))
        corr_mat = cosine_similarity(gen_)
        # sort correlation value ascendingly and select top_k item_id
        top_items = corr_mat[item_id, :].argsort()[-top_k:][::-1]
        top_items = [map_name[e] for e in top_items]

        print(movies1.loc[movies1['movieId'].isin(top_items)])
        print('\n')
```

### Function : convert_mov_and_idx(gen)
#### The id value of the movie corresponding to the index of the data frame is stored, and the index of the data frame corresponding to the id value of the movie is stored.
##### Parameter:
##### gen | dataframe
##### Return:
##### i2m | dict
##### m2i | dict
```python
def convert_mov_and_idx(gen):
    i2m = {index: movie for index, movie in enumerate(gen.index)}
    m2i = {v: k for k, v in i2m.items()}
    return i2m, m2i
```

### Function : read_contentBased_test()
#### Read a test dataset consisting of movie id
##### Return:
##### Int_test_list | list
```python
def read_contentBased_test():
    df = pd.read_csv('input/contentBased_knn_test.csv')
    test_list = df.columns.values.tolist()
    int_test_list = list(map(int, test_list))

    return int_test_list
```



## Content-Based Filtering using TF-IDF vectorizer
reference: 
1. https://www.kaggle.com/code/sankha1998/content-based-movie-reommendation-system

### Function : preprocessing_tfidf
#### get movie, rating data, and Data Transformation
##### Parameter: 
##### movie   |  path of movie.csv
##### rating  |  path of rating.csv
##### Return:
##### df      |  data frame that transformed
##### sigmoid matrix | TFIDF vectorized matrix using sigmoid kernel
##### cosine matrix  | TFIDF vectorized matrix using cosine similarity
                 
```python
def preprocessing_tfidf(movie,rating):
    movie=pd.read_csv(movie)
    rating=pd.read_csv(rating)

    # movie and rating are sutable for analysis
    movie_details=movie.merge(rating,on='movieId')

    movie_details.head()

    movie_details.drop(columns=['timestamp'],inplace=True)
    total_ratings=movie_details.groupby(['movieId','genres']).sum()['rating'].reset_index()
    df=movie_details.copy()
    df.drop_duplicates(['title','genres'],inplace=True)
    df=df.merge(total_ratings,on='movieId')
    df.drop(columns=['userId','rating_x','genres_y'],inplace=True)
    df.rename(columns={'genres_x':'genres','rating_y':'rating'},inplace=True)
    df.head()

    df['rating']=df['rating'].astype(int)

    print(df['genres'].value_counts())
    #TFIDF vectorizer
    tfv = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1)
    tfidf_matrix= tfv.fit_transform(df['genres'])

    #matrics using sigmoid kernel
    sigmoid_matrix = sigmoid_kernel(tfidf_matrix,tfidf_matrix)
    print('\nSimilarity Metrics(sigmoid kernel)')
    print(sigmoid_kernel)
    # matrics using cosine similarity
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print('\nSimilarity Metrics(cosine similarity)')
    print(cosine_matrix)
    df1=df.copy()
    ti=[]
    for i in df1['title']:
        ti.append(i.split(' (')[0])
    df1['title']=ti
    return df1, sigmoid_matrix,cosine_matrix
```

### Function : recommendations_cosine_similarity
#### Content-based recommendation using cosine similarity
##### Parameter: 
##### title   |  Query movie title to find recommendations
#####             df1  |  data frame that transformed before
#####             cosine_matrix | TFIDF vectorized matrix using cosine similarity

```python
def recommendations_cosine_similarity(title,df1,cosine_matrix):
    i_d = []
    indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()
    idx = indices[title]
    dis_scores = list(enumerate(cosine_matrix[idx]))
    dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)
    dis_scores = dis_scores[1:31]
    idn = [i[0] for i in dis_scores]
    final = df1.iloc[idn].reset_index()
    idn = [i for i in final['index']]
    for j in idn:
        if (j < 15951):
            i_d.append(j)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    print('\ncontent-based-recommendation using TF-IDF (cosine similarity)')
    print('query : '+ title)
    for i in range(1, 8):
        if (idn):
            score = np.round(dis_scores[i][1], 6)
            print(('{},  {}'.format(indices.iloc[i_d].index[i], score)))
```

### Function : recommendations_sigmoid_kernel
#### Content-based recommendation using cosine similarity
##### Parameter:
##### title   |  Query movie title to find recommendations
##### df1  |  data frame that transformed before
##### sigmoid_matrix | TFIDF vectorized matrix using sigmoid_matrix

```python
def recommendations_sigmoid_kernel(title,df1,sigmoid_matrix):
    i_d = []
    indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()
    idx = indices[title]
    dis_scores = list(enumerate(sigmoid_matrix[idx]))
    dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)
    dis_scores = dis_scores[1:31]

    idn = [i[0] for i in dis_scores]
    final = df1.iloc[idn].reset_index()
    idn = [i for i in final['index']]
    for j in idn:
        if (j < 15951):
            i_d.append(j)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    print('\ncontent-based-recommendation using TF-IDF(sigmoid kernel):')
    print('query : ' + title)
    for i in range(1, 8):
        if (idn):
            score = np.round( dis_scores[i][1],6)
            print(('{},  {}'.format(indices.iloc[i_d].index[i], score)))
```

## Collaborative Filtering using KNN
reference:
1. https://pearlluck.tistory.com/667


### Function : collaborative_preprocessing(movies, ratings)
#### It receives movie and rating datasets as input. Modify the ratings dataset to only consider users who have rated more than 80 movies. Delete unnecessary columns.
#####	Parameter:
#####	movies | dataframe
#####	ratings | dataframe
#####  Return:
#####  ratings_t1 | dataframe

```python
def collaborative_preprocessing(movies):
    movies['genres'] = movies['genres'].str.replace('|', ' ')

    # limit ratings to user ratings that have rated more than 80 movies
    # Otherwise it becomes impossible to pivot the rating dataframe later for collaborative filtering
    ratings_t = ratings.groupby('userId').filter(lambda x: len(x) > 80)

    # list the movie titles that survive the filtering
    movie_list_rating = ratings_t.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]

    ratings_t.drop(['timestamp'], axis=1, inplace=True)
    ratings_t = ratings_t.iloc[:1000000, :]
    ratings_t1 = pd.merge(movies[['movieId']], ratings_t, on='movieId', how='right')

    print(ratings_t1.head(5))

    return ratings_t1
```

### Function : collaborative_model_building(my_ratings)
#### Create a table with index=movie, column=user, and treat NaN as 0. Convert this table to CSR matrix and save it. Generate a knn model by putting this matrix as data.
##### Parameter:
##### my_ratings | dataframe
##### Return:
##### model_knn | model
##### mat_movies_users | csr_matrix

```python
def collaborative_model_building(my_ratings):
    movies_users = my_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(mat_movies_users)

    return model_knn, mat_movies_users
```
### Function : collaborative_recommender(test_list, data, model, n_recommendations)
#### As a function that recommends movies with similar ratings to the movies entered as input. The similarity is calculated using the cosine_similarity, and print the information of the movie with the highest similarity.
##### Parameter:
##### test_list | list
#####	data | dataframe
##### model | model
##### n_recommendations | int

```python
def collaborative_recommender(test_list, data, model, n_recommendations):
    model.fit(data)

    for movie_name in test_list:
        idx = process.extractOne(movie_name, movies['title'])[2]
        print('Movie Selected: ', movies['title'][idx], 'Index: ', idx)
        print('Searching for recommendations.....')
        distances, indices = model.kneighbors(data[idx], n_neighbors=n_recommendations)
        for i in indices:
            print(movies['title'][i].where(i != idx))
        print('\n')
```

### Function : read_collaborative_test()
#### Read a test dataset consisting of movie titles
##### Return:
##### test_list | list
```python
def read_collaborative_test():
    df = pd.read_csv('input/collaborative_knn_test.csv')
    test_list = df.columns.values.tolist()
    return test_list
```



## Collaborative Filtering using SGD
reference: https://yamalab.tistory.com/92


### Function : preprocessing_gradient()
#### Read input file data and Make uid x movie matrix


```python
def preprocessing_gradient():
    movie = pd.read_csv("input/movie.csv")
    ratings = pd.read_csv("input/rating.csv")

    data = pd.merge(left=ratings, right=movie, on='movieId', how='left')

    data = data.iloc[:100000, :]
    movie_user = data.pivot_table(index='userId', columns='title', values='rating')

    movie_user = movie_user.fillna(0)
    np_movie_user = movie_user.to_numpy()

    return movie_user, np_movie_user
```


### Function : writing_gradient(before, after, recommend)
#### Make before, after, recommend dataframe csv files

```python
def writing_gradient(before, after, recommend):
    before.to_csv("before_g.csv", mode='w')
    after.to_csv("after_g.csv", mode='w')
    recommend.to_csv("recommend_g.csv", mode='w')
```
### class MatrixFactorization()
#### Function : def _init_(self, R, k, learning_rate, reg_param, epochs, verbose=False)
##### This function runs automatically when the class is called.
###### Parameter:
###### R: rating matrix
###### k: latent parameter
###### learning_rate: alpha on weight update
###### reg_param: beta on weight update
###### epochs: training epochs
###### verbose: print status

#### Function : fit(self)
Train Matrix factorization with updating matrix latent weight and bias

#### Function : cost(self)
Calculate the error for the entire matrix.

#### Function : gradient(self, error, i, j)
Calculate gradient of latent feature for gradient descent

#### Function : gradient_descent(self, i, j, rating)
Do gradient descent

#### Function : get_prediction(self, i, j)
Get predicted ratings

#### Function : get_complete_matrix(self)
Make complete rating matrix


```python
class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self):

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)


    def gradient(self, error, i, j):

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):

        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):

        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


    def print_results(self):

        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        R_matrix = self.get_complete_matrix()
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])
        return R_matrix
```






## Main Code 
###  Content-Based Filtering, Collaborative Filtering , File IO 
```python
# Content-based filtering
#TFIDF vectorizer
df,sigmoid_matrix,cosine_matrix = preprocessing_tfidf('input/movie.csv','input/rating.csv')
recommendations_sigmoid_kernel('Sunset',df,sigmoid_matrix)
recommendations_cosine_similarity('Sunset',df,cosine_matrix)
#Knn
my_genres = content_based_preprocessing(movies)
ind2mov, mov2ind = convert_mov_and_idx(my_genres)
# Read the content-based test dataset
contentBased_test_list = read_contentBased_test()
print('\n***** Test List for Content-Based')
print(contentBased_test_list)
print('\n')
content_based_model_build(gen_= my_genres, my_list = contentBased_test_list, top_k=10, map_name = ind2mov)



# Collaborative filtering
my_ratings = collaborative_preprocessing(movies)
my_model, my_movie_user = collaborative_model_building(my_ratings)
# Read the collaborative test dataset
collaborative_test_list = read_collaborative_test()
print('\n***** Test List for Collaborative')
print(collaborative_test_list)
print('\n')
collaborative_recommender(collaborative_test_list, my_movie_user, my_model, 10)
#SGD
movie_user_g, np_movie_user_g = preprocessing_gradient()
R = np_movie_user_g
factorizer = MatrixFactorization(R, k=3, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
factorizer.fit()
R_matrix_g = factorizer.print_results()

movie_columns_g = movie_user_g.columns
df_r_matrix_g = pd.DataFrame(R_matrix_g)
df_r_matrix_g.columns = movie_columns_g
df_r_matrix_g = df_r_matrix_g.set_index(movie_user_g.index)

recommend_g = np.where(R_matrix_g.astype(float) > 3.5, 1, 0)
df_recommend_g = pd.DataFrame(recommend_g)
df_recommend_g.columns = movie_columns_g
df_recommend_g = df_recommend_g.set_index(movie_user_g.index)

writing_gradient(movie_user_g, df_r_matrix_g, df_recommend_g)
```


# Results

