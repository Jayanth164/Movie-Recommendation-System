import os
import logging
from pathlib import Path

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


# [PRODUCTION ADDITION] Configure logging instead of relying only on print statements.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# [PRODUCTION ADDITION] Use an environment variable or configurable path.
# Set MOVIES_CSV_PATH if the CSV is stored somewhere else.
movies_csv_path = Path(
    os.getenv(
        "MOVIES_CSV_PATH",
        "/Users/nissankararaojayanth/Desktop/Movie Recommendation System/recommendation/movies.csv"
    )
)


# [PRODUCTION ADDITION] Define the expected schema in one place.
required_columns = {
    "index",
    "title",
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director"
}


#Data Collection and Pre-processing
if not movies_csv_path.is_file():
    raise FileNotFoundError(
        f"Movie dataset was not found at: {movies_csv_path}"
    )


movies_data=pd.read_csv(movies_csv_path)

# [PRODUCTION ADDITION] Validate the input schema immediately.
missing_columns = required_columns - set(movies_data.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_columns)}"
    )

if movies_data.empty:
    raise ValueError("The movie dataset is empty.")

# [PRODUCTION ADDITION] Validate title values before recommendation logic.
if movies_data["title"].isna().all():
    raise ValueError("The dataset does not contain any usable movie titles.")

# [PRODUCTION ADDITION] Normalize the title column to avoid matching problems.
movies_data["title"] = movies_data["title"].fillna("").astype(str).str.strip()

# [PRODUCTION ADDITION] Remove rows that cannot produce valid recommendations.
movies_data = movies_data[movies_data["title"] != ""].copy()

if movies_data.empty:
    raise ValueError("No rows with valid movie titles remain after cleaning.")

# [PRODUCTION ADDITION] Ensure the row position and dataset index are not confused.
# The similarity matrix uses row positions, not necessarily values in the "index" column.
movies_data = movies_data.reset_index(drop=True)


#Print the first 5 columns in the data
print(movies_data.head())


#Number of rows and columns in data frame
print(movies_data.shape)


#Feature Extraction
#Selecting the relevant features for recommendation
selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)


# [PRODUCTION ADDITION] Validate feature columns before using them.
missing_feature_columns = set(selected_features) - set(movies_data.columns)

if missing_feature_columns:
    raise ValueError(
        f"Selected feature columns are missing: "
        f"{sorted(missing_feature_columns)}"
    )


#Replacing the null values with null string
#For every time this "for loop" runs each time it will genres, second time keywords, third time tagline like this and will replace null values 
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna(' ')

    # [PRODUCTION ADDITION] Normalize values before combining them.
    movies_data[feature] = (
        movies_data[feature]
        .astype(str)
        .str.strip()
    )


#Combine all the 5 selected features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)


# [PRODUCTION ADDITION] Ensure every combined record is a string.
combined_features = combined_features.fillna("").astype(str)


#Converting the text data into feature vectors
vectorizer=TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    dtype=np.float32
)


#The textual data in the feature_vectors will be converted into the numerical form
feature_vectors=vectorizer.fit_transform(combined_features )
print(feature_vectors)


#Getting the similarity scores
#Store the similarity scores of all the movies
# [PRODUCTION FIX] Do not calculate the complete N x N matrix for every request.
# It is expensive in memory for a large dataset.
print("Feature matrix shape:", feature_vectors.shape)


# [PRODUCTION ADDITION] Create a normalized title lookup for robust matching.
normalized_titles = (
    movies_data["title"]
    .str.casefold()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

title_to_row_position = {}

for row_position, normalized_title in enumerate(normalized_titles):
    title_to_row_position.setdefault(normalized_title, row_position)


#Getting the movie name from the user
movie_name=input("Enter your favourite movie: ").strip()


if not movie_name:
    raise ValueError("Movie name cannot be empty.")


#After getting the movie name from the user, now we need to write code to see what movies are closer to that movie
#Create a list with all movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)


#Finding the closely matched movie name to that of user given movie name
find_close_match=difflib.get_close_matches(
    movie_name,
    list_of_all_titles,
    n=3,
    cutoff=0.6
)
print(find_close_match)
#The movies which are close to the movie name given by user will be stored in "find_close_match" variable


if not find_close_match:
    print(
        "No close movie title was found. "
        "Please enter a title closer to one in the dataset."
    )
    raise SystemExit(0)


#Now we will print the first movie name in the "find_close_match" variable
close_match=find_close_match[0]
print(close_match)


#Finding the index of movie with title
# [PRODUCTION FIX] Use the DataFrame row position for the similarity matrix.
# The dataset's "index" column may not equal the matrix row position.
index_of_the_movie = title_to_row_position[
    close_match.casefold().strip()
]
print(index_of_the_movie)


#Getting a list of similar movies
#Now we will get the output of index number and the similarity score of that movie to that of user given movie
#This is not for only the movies which are similar to user given movie
#This gives the output of all movies index number and the similarity score to that of user given movie

# [PRODUCTION FIX] Calculate similarity only between the selected movie and all movies.
similarity_score = cosine_similarity(
    feature_vectors[index_of_the_movie],
    feature_vectors
).ravel()

# [PRODUCTION ADDITION] Exclude the selected movie from its own recommendations.
similarity_score[index_of_the_movie] = -1.0


#Now we will sort the movies based on their similarity score
sorted_similar_movies=sorted(
    enumerate(similarity_score),
    key=lambda x:x[1],
    reverse=True
)
print(sorted_similar_movies)
#We have put the parameter reverse=True because we are arranging movies from last(Movies which are having highest similarity score)
#In the variable "X" we will store the similarity_score
#x[1] means the movie name


#Print the name of the similar movies based on the index
print("Movies suggested for you: \n")


i=1
recommendation_limit = 30

for movie in sorted_similar_movies:
    index=movie[0] #We are fetching index of movie name and storing it in variable named "index"
    title_from_index=movies_data.iloc[index]["title"]

    if(i <= recommendation_limit): #Only first 30 movies
        print(i,'.',title_from_index)
        i+=1

    else:
        break