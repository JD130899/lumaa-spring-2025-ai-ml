Overview
This project implements a simple content-based recommendation system using the IMDB 5000 Movie Dataset. It takes a short text description of user preferences as input and returns the top 5 recommended movies based on textual similarity. The system leverages TF-IDF vectorization and cosine similarity to compare the user’s input with movie descriptions created from the dataset's genres and plot keywords.

Dataset
Dataset Name: IMDB 5000 Movie Dataset
Source: Available on Kaggle or other public repositories.
Description:
This dataset includes metadata for 5000 movies. The CSV file should contain at least the columns movie_title, genres, and plot_keywords. The project combines genres and plot_keywords to form a description for each movie.
Setup:
Download the dataset (typically named movie_metadata.csv) and place it in your project directory.
Setup
Python Version: Python 3.7 or higher

Running the Code
Run the recommendation script by passing your query as a command-line argument. For example:
python /Users/jaipdalvi/Desktop/Work/Python\ DSA/9/demo.py "I love thrilling action movies set in space, with a comedic twist."

The script will output the top 5 recommended movies along with their similarity scores.

Results
Here is an example output for the above query:
1. Urban Legends: Final Cut  (Score: 0.368)
2. Cargo  (Score: 0.293)
3. Space Cowboys  (Score: 0.270)
4. Harlock: Space Pirate  (Score: 0.241)
5. Space Dogs  (Score: 0.240)

This output indicates that the system finds movies like Space Cowboys and Harlock: Space Pirate relevant to your query, based on the combined genre and keyword information.

Additional Information
Salary Expectation per Month: $6400/month
Demo Video: Please refer to the demo.md file for a link to a screen recording demonstrating how to run the code and view the results.