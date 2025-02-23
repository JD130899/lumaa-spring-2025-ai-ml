import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the IMDB 5000 Movie Dataset from a CSV file.
    The CSV should include the columns 'movie_title', 'genres', and 'plot_keywords'.
    A new 'description' column is created by combining genres and plot keywords.
    """
    df = pd.read_csv(path)
    # Fill missing values and create a description column
    df['genres'] = df['genres'].fillna('')
    df['plot_keywords'] = df['plot_keywords'].fillna('')
    df['description'] = "Genres: " + df['genres'] + ". Keywords: " + df['plot_keywords']
    return df

def build_tfidf_matrix(descriptions: pd.Series):
    """
    Convert text descriptions to TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return tfidf_matrix, vectorizer

def get_recommendations(query: str, tfidf_matrix, vectorizer, df: pd.DataFrame, top_n: int = 5):
    """
    Compute cosine similarity between the query and all item descriptions,
    and return the top_n recommendations.
    """
    # Transform the query into the TF-IDF vector space
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarities between query and each description
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of top_n similar items
    indices = cosine_sim.argsort()[::-1][:top_n]
    
    # Return the recommended items and their similarity scores
    recommended_items = df.iloc[indices]
    scores = cosine_sim[indices]
    return recommended_items, scores

def main():
    if len(sys.argv) < 2:
        print("Usage: python recommend.py 'Your query text here'")
        sys.exit(1)
    
    query = sys.argv[1]
    dataset_path = '/Users/jaipdalvi/Desktop/Work/Python DSA/9/movie_metadata.csv'  # Download the IMDB 5000 dataset from Kaggle and place it here.
    
    # Load and process the dataset
    df = load_dataset(dataset_path)
    tfidf_matrix, vectorizer = build_tfidf_matrix(df['description'])
    
    # Get recommendations
    recommendations, scores = get_recommendations(query, tfidf_matrix, vectorizer, df)
    
    # Print out the results
    print("Top Recommendations:")
    for i, (idx, row) in enumerate(recommendations.iterrows()):
        print(f"{i+1}. {row['movie_title']} (Score: {scores[i]:.3f})")

if __name__ == '__main__':
    main()
