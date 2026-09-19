import numpy as np 
#Used to create numpy arrays

import pandas as pd 
#Used to create data frames, because we are having the data in the csv format, which is not in structured format

import difflib
#When we ask the user to give his fav movie name, he might give the movie name with spelling mistakes
#So using this library we can find the movie name, which is closest to the name given by the user.

from sklearn.feature_extraction.text import TfidfVectorizer
#To convert the textual data into the numerical form

from sklearn.metrics.pairwise import cosine_similarity
#Used to find the similarity score. When the user gives his fav movie name we will find the movies which are similar to that movie


#Data Collection and Pre-processing
movies_data=pd.read_csv('/Users/nissankararaojayanth/Desktop/Movie Recommendation System/recommendation/movies.csv')


#Print the first 5 columns in the data
print(movies_data.head())

#Number of rows and columns in data frame
print(movies_data.shape)

#Feature Extraction
#Selecting the relevant features for recommendation
selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)

#Replacing the null values with null string
#For every time this "for loop" runs each time it will genres, second time keywords, third time tagline like this and will replace null values 
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna(' ')

#Combine all the 5 selected features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)

#Converting the text data into feature vectors
vectorizer=TfidfVectorizer()

#The textual data in the feature_vectors will be converted into the numerical form
feature_vectors=vectorizer.fit_transform(combined_features )
print(feature_vectors)

#Getting the similarity scores
#Store the similarit scores of all the movies
similarity=cosine_similarity(feature_vectors)
print(similarity)

#Used to print the movie index and similarity scores as well
print(similarity.shape)

#Getting the movie name from the user
movie_name=input("Enter your favourite movie: ")

#After getting the movie name from the user, now we need to write code to see what movies are closer to that movie
#Create a list with all movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)

#Finding the closely matched movie name to that of user given movie name
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)
#The movies which are close to the movie name given by user will be stored in "find_close_match" variable

#Now we will print the first movie name in the "find_close_match" variable
close_match=find_close_match[0]
print(close_match)

#Finding the index of movie with title
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie) 

#Getting a list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)
#Now we will get the output of index number and the similarity score of that movie to that of user given his fav movie
#This is not for only the movies which are similar to user given movie
#This gives the output of all movies index number and their similarity score to that of user given movie

print(len(similarity_score))
#Now we will get how many movies are there(Because similarity score contains score for all the existing movies in the data)

#Now we will sort the movies based on their similarity score
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
print(sorted_similar_movies)
#We have put the parameter reverse=True because we are arranging movies from last(Movies which are having highest similarity score)
#In the variable "X" we will store the similarity_score
#x[1] means the movie name

#Print the name of the similar movies based on the index
print("Movies suggested for you: \n")

i=1
for movie in sorted_similar_movies:
    index=movie[0] #We are fetching index of movie name and storing it in variable named "index"
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if(i<30): #Only first 30 movies
        print(i,'.',title_from_index)
        i+=1