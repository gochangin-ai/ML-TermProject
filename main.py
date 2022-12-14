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

"""
reference : https://www.kaggle.com/code/sankha1998/content-based-movie-reommendation-system
content-based filtering using TF-IDF vectorizer
"""
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




"""
reference: https://yamalab.tistory.com/92
collaborative filtering using SGD
"""
def preprocessing_gradient():
    movie = pd.read_csv("input/movie.csv")
    ratings = pd.read_csv("input/rating.csv")

    data = pd.merge(left=ratings, right=movie, on='movieId', how='left')

    data = data.iloc[:100000, :]
    movie_user = data.pivot_table(index='userId', columns='title', values='rating')

    movie_user = movie_user.fillna(0)
    np_movie_user = movie_user.to_numpy()

    return movie_user, np_movie_user

def writing_gradient(before, after, recommend):
    before.to_csv("before_g.csv", mode='w')
    after.to_csv("after_g.csv", mode='w')
    recommend.to_csv("recommend_g.csv", mode='w')

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

            # rating??? ???????????? index??? ???????????? training
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

        # xi, yi: R[xi, yi]??? nonzero??? value??? ????????????.
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

"""
collaborative filtering & content-based filtering using KNN 
References
1. https://pearlluck.tistory.com/667
2. https://www.kaggle.com/code/yogeshrampariya/content-based-recommendation-system
"""
def collaborative_preprocessing(movies, ratings):
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

def collaborative_model_building(my_ratings):
    movies_users = my_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model_knn.fit(mat_movies_users)

    return model_knn, mat_movies_users



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


def read_collaborative_test():
    df = pd.read_csv('input/collaborative_knn_test.csv')
    test_list = df.columns.values.tolist()
    return test_list

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

def List_All_Genres(movies):
    genres = []
    for genre in movies.genres:
        x = genre.split('|')
        for i in x:
            if (str(i).lower() not in genres):
                my_genres = str(i).lower()
                genres.append(my_genres)
    return genres

def content_based_model_build(gen_, my_list, top_k, map_name):
    for item_id in my_list:
        print('Movie Selected : {0}'.format(movies1.values[item_id]))
        corr_mat = cosine_similarity(gen_)
        # sort correlation value ascendingly and select top_k item_id
        top_items = corr_mat[item_id, :].argsort()[-top_k:][::-1]
        top_items = [map_name[e] for e in top_items]

        print(movies1.loc[movies1['movieId'].isin(top_items)])
        print('\n')

def convert_mov_and_idx(gen):
    i2m = {index: movie for index, movie in enumerate(gen.index)}
    m2i = {v: k for k, v in i2m.items()}
    return i2m, m2i

def read_contentBased_test():
    df = pd.read_csv('input/contentBased_knn_test.csv')
    test_list = df.columns.values.tolist()
    int_test_list = list(map(int, test_list))

    return int_test_list


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
my_ratings = collaborative_preprocessing(movies, ratings)
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
