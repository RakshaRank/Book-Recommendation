import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

st.title("Book Recommender")

st.write("This is a simple book recommender app")

#how to load csv in streamlit app

#dataset_name = st.sidebar.selectbox("Select Dataset", ("Data","Books"))
classifier_name = st.sidebar.selectbox("Select Classifer", ("Popularity","KNN"))

data = pd.read_csv("cleaned_data.csv")
books = pd.read_csv("books.csv", low_memory=False)
ratings_data = pd.read_csv("ratings_data.csv")


n = st.sidebar.selectbox("How many books you want? ", ("1","2","3","4","5","6","7","8","9"))
n = int(n)

def filtering_on_popularity(df, n):
    if n >= 1 and n <= len(df):
        data = pd.DataFrame(df.groupby('ISBN')['Book-Rating'].count()).sort_values('Book-Rating', ascending=False).head(n)
        result = pd.merge(data, books, on='ISBN')
        return result
    return "Something went wrong!"
    print("Top", n, "most popular books: ")

def filtering_on_knn(df, n, title):

    # limit_popularity = 30

    # count_user = ratings_data['User-ID'].value_counts()
    # df = ratings_data[ratings_data['User-ID'].isin(count_user[count_user >= limit_popularity].index)]
    # count_ratings = df['Book-Rating'].value_counts()
    # df = df[df['Book-Rating'].isin(count_ratings[count_ratings >= limit_popularity].index)]

    # df_2 = df.pivot_table(index='Book-Title', columns='User-ID', values = 'Book-Rating').fillna(0)
    # df_2.to_csv('df_2.csv')
    # df_2 = pd.read_csv('df_2.csv')
    # df_2.set_index('Book-Title', inplace=True)


    # book_sparse = csr_matrix(df_2)

    
    # model = NearestNeighbors(algorithm='brute',n_neighbors=n)
    # model.fit(book_sparse)

    
    # filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))


    #get the title from the user as string
    df_2 = pd.read_csv('df_2.csv')
    df_2.set_index('Book-Title', inplace=True)
    model = pickle.load(open('model.sav', 'rb'))
   
    #find the title in the dataset and return the index
    index = df_2.index.get_loc(title)

    _, suggestions = model.kneighbors(df_2.iloc[index, :].values.reshape(1, -1))

    sugg_names = [df_2.index[suggestions[0][i+1]] for i in range(n)]
    
    return pd.DataFrame(data = {'Recommendations':sugg_names},index = range(1,n+1))


def classify(classifier_name, data):

    if classifier_name == "Popularity":
        st.write("Top", n, "most popular books: ")
        st.write(filtering_on_popularity(data, int(n)))  
    elif classifier_name == "KNN":
        st.write("KNN")
        title = st.text_input("Enter the book title: ")
        if title != '':
            st.write(filtering_on_knn(data, int(n), title))
    else:
        st.write("Something went wrong!")

classify(classifier_name, data)     
        
