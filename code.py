import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import random
import pickle
import os
from datetime import datetime
import numpy as np

# GPU/CUDA imports
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🚀 GPU (CUDA) detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, falling back to CPU")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not found. Install with: pip install torch")
    print("    Falling back to CPU mode...")

# --- Configuration ---
CSV_FILE_PATH = "/mnt/c/Users/abhin/OneDrive/Desktop/Ironfleet/Archives/Content-Based Recommender system/archive/TMDB_movie_dataset_v11.csv"
CACHE_FILE = "movie_recommender_cache_gpu.pkl"

print("\n--- Step 1: Loading Data from Local File ---")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please make sure the CSV file is in the same folder as this script.")
    sys.exit()

# --- Step 2: Data Cleaning & Date Handling ---
print("--- Step 2: Cleaning & Filtering ---")

# Convert release_date to actual date objects
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Extract the year for easier display
df['year'] = df['release_date'].dt.year.fillna(0).astype(int)

# Filter for English movies
df = df[df['original_language'] == 'en']

# Keep ALL movies - don't limit to top 20,000
print(f"Total movies in dataset: {len(df):,}")

# Reset index and fill missing data
df = df.reset_index(drop=True)
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('Unknown')

# Handle ratings - fill missing values with median
if 'vote_average' in df.columns:
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].median())
else:
    df['vote_average'] = 0

if 'vote_count' in df.columns:
    df['vote_count'] = df['vote_count'].fillna(0)
else:
    df['vote_count'] = 0

# --- Calculate Weighted Rating (IMDB Formula) ---
C = df['vote_average'].median()
m = df['vote_count'].quantile(0.60)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    if v + m == 0:
        return C
    return (v / (v + m) * R) + (m / (v + m) * C)

df['weighted_rating'] = df.apply(weighted_rating, axis=1)

# Normalize weighted rating to 0-1 scale
rating_range = df['weighted_rating'].max() - df['weighted_rating'].min()
if rating_range > 0:
    df['rating_score'] = (df['weighted_rating'] - df['weighted_rating'].min()) / rating_range
else:
    df['rating_score'] = 0.5

# Create the "Soup" for content-based filtering
df['tags'] = df['overview'] + " " + df['genres']

print(f"Modeling on {len(df):,} total movies (Released & Upcoming).")

# --- Step 3: Training the Model with GPU Support ---
print("\n--- Step 3: Training/Loading Model ---")

def cosine_similarity_gpu(query_vector, all_vectors, batch_size=2500):
    """
    Memory-efficient GPU cosine similarity optimized for 6GB VRAM.
    Works with sparse matrices without converting entire matrix to dense.
    
    Memory calculation for batch_size=2500:
    - Batch: 2500 x 3000 x 4 bytes (float32) = ~30MB
    - Query: 1 x 3000 x 4 bytes = ~12KB
    - Computation overhead: ~50MB
    Total per iteration: ~80MB (well within 6GB VRAM)
    """
    if GPU_AVAILABLE:
        device = torch.device('cuda')
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        
        # Convert query vector to dense (single vector, safe)
        query_dense = query_vector.toarray().astype(np.float32)
        query_tensor = torch.from_numpy(query_dense).to(device)
        query_norm = torch.norm(query_tensor, dim=1, keepdim=True)
        query_normalized = query_tensor / (query_norm + 1e-10)
        
        n_samples = all_vectors.shape[0]
        similarities = np.zeros(n_samples, dtype=np.float32)
        
        # Calculate optimal batch size based on available VRAM
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory * 0.7  # Use 70% of VRAM
            # Each sample: 3000 features * 4 bytes = 12KB
            optimal_batch = int(free_memory / (3000 * 4 * 2))  # *2 for safety margin
            batch_size = min(batch_size, optimal_batch)
        
        print(f"⚡ Processing {n_samples:,} movies in batches of {batch_size:,}...")
        print(f"   Estimated time: ~{(n_samples / batch_size) * 0.05:.1f}s")
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Convert ONLY this batch to dense (memory-safe)
            batch_sparse = all_vectors[start_idx:end_idx]
            batch_dense = batch_sparse.toarray().astype(np.float32)
            batch_tensor = torch.from_numpy(batch_dense).to(device)
            
            # Normalize batch
            batch_norm = torch.norm(batch_tensor, dim=1, keepdim=True)
            batch_normalized = batch_tensor / (batch_norm + 1e-10)
            
            # Compute cosine similarity for this batch
            batch_similarities = torch.mm(query_normalized, batch_normalized.T)
            similarities[start_idx:end_idx] = batch_similarities.cpu().numpy().flatten()
            
            # Aggressive GPU memory cleanup
            del batch_dense, batch_tensor, batch_normalized, batch_norm, batch_similarities
            torch.cuda.empty_cache()
            
            # Progress indicator every 10 batches
            if (start_idx // batch_size) % 10 == 0 or end_idx == n_samples:
                progress = (end_idx / n_samples) * 100
                print(f"   Progress: {progress:.1f}% ({end_idx:,}/{n_samples:,}) | "
                      f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", end='\r')
        
        # Final cleanup
        del query_tensor, query_normalized
        torch.cuda.empty_cache()
        
        print()  # New line after progress
        return similarities
    else:
        # CPU fallback using sklearn (handles sparse matrices efficiently)
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(query_vector, all_vectors).flatten()

def train_model():
    """Train the content-based model using TfidfVectorizer"""
    print("Training new model (this may take a moment)...")
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', 
                            min_df=2, max_df=0.8, ngram_range=(1, 2))
    vector = tfidf.fit_transform(df['tags'])
    
    print(f"✓ Vectorized {len(df):,} movies")
    print(f"✓ Vector shape: {vector.shape} (sparse matrix)")
    if GPU_AVAILABLE:
        print(f"✓ GPU acceleration enabled for similarity calculations")
    return vector, tfidf

def save_cache(vector_matrix, vectorizer):
    """Save the vector matrix and vectorizer to disk"""
    cache_data = {
        'vectors': vector_matrix,
        'vectorizer': vectorizer,
        'df_titles': df['title'].tolist(),
        'df_index': df.index.tolist(),
        'timestamp': datetime.now(),
        'gpu_enabled': GPU_AVAILABLE
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"✓ Cache saved to {CACHE_FILE}")

def load_cache():
    """Load cached vectors if available"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache matches current dataframe
            if (cache_data['df_titles'] == df['title'].tolist() and 
                cache_data['df_index'] == df.index.tolist()):
                cached_time = cache_data['timestamp'].strftime('%Y-%m-%d %H:%M')
                gpu_status = "GPU" if cache_data.get('gpu_enabled', False) else "CPU"
                print(f"✓ Loaded cached model from {cached_time} [{gpu_status}]")
                return cache_data['vectors'], cache_data['vectorizer']
            else:
                print("⚠ Cache doesn't match current data, retraining...")
                return None, None
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
            return None, None
    return None, None

# Try to load from cache, otherwise train new model
vectors, vectorizer = load_cache()
if vectors is None:
    vectors, vectorizer = train_model()
    save_cache(vectors, vectorizer)

print("Model Ready!")

# --- Hybrid Recommendation Function with GPU ---
def hybrid_recommend(movie_name, content_weight=0.7, rating_weight=0.3, num_recommendations=5):
    """
    Hybrid recommendation with GPU-accelerated similarity calculation
    """
    movie_name = movie_name.strip().lower()
    titles_lower = df['title'].str.lower().values

    if movie_name not in titles_lower:
        print(f"\n❌ Result: '{movie_name}' not found in the dataset.")
        # Find close matches
        from difflib import get_close_matches
        close = get_close_matches(movie_name, df['title'].str.lower().tolist(), n=5, cutoff=0.6)
        if close:
            print("\n💡 Did you mean one of these?")
            for match in close:
                idx = list(titles_lower).index(match)
                print(f"  • {df.iloc[idx].title} ({df.iloc[idx].year})")
        return

    # Find the movie index
    index = list(titles_lower).index(movie_name)
    
    mode = "GPU (CUDA)" if GPU_AVAILABLE else "CPU"
    print(f"\n⏳ Searching through {len(df):,} movies using {mode}...")
    
    # Calculate similarity using GPU if available
    import time
    start_time = time.time()
    
    movie_vector = vectors[index]
    content_scores = cosine_similarity_gpu(movie_vector, vectors)
    
    elapsed = time.time() - start_time
    print(f"⚡ Similarity calculated in {elapsed:.2f} seconds")
    
    # Normalize content scores to 0-1 range
    content_range = content_scores.max() - content_scores.min()
    if content_range > 0:
        content_scores_norm = (content_scores - content_scores.min()) / content_range
    else:
        content_scores_norm = content_scores
    
    # Get rating scores
    rating_scores = df['rating_score'].values
    
    # HYBRID SCORING: Combine content similarity with rating quality
    hybrid_scores = (content_weight * content_scores_norm) + (rating_weight * rating_scores)
    
    # Get top candidates
    similar_movies = sorted(list(enumerate(hybrid_scores)), reverse=True, key=lambda x: x[1])
    
    # Skip the first one (the movie itself) and take top 25
    top_candidates = similar_movies[1:26]
    
    # Randomly select from top 25 for variety
    if len(top_candidates) >= num_recommendations:
        selected_movies = random.sample(top_candidates, num_recommendations)
    else:
        selected_movies = top_candidates

    # Display results
    input_movie = df.iloc[index]
    print(f"\n{'='*60}")
    print(f"🎬 Recommendations for '{input_movie.title}' ({input_movie.year})")
    print(f"   Rating: {input_movie['vote_average']:.1f}/10 ({int(input_movie['vote_count']):,} votes)")
    print(f"   Searched: {len(df):,} movies")
    print(f"{'='*60}")
    
    today = datetime.now()

    for idx, (movie_idx, score) in enumerate(selected_movies, 1):
        title = df.iloc[movie_idx].title
        genres = df.iloc[movie_idx].genres
        release_date = df.iloc[movie_idx].release_date
        year = df.iloc[movie_idx].year
        rating = df.iloc[movie_idx]['vote_average']
        votes = int(df.iloc[movie_idx]['vote_count'])
        
        # Check if upcoming
        status_tag = ""
        if pd.notnull(release_date) and release_date > today:
            status_tag = " 🔜 [UPCOMING]"
        
        # Display with rating info
        print(f"\n{idx}. {title} ({year}){status_tag}")
        print(f"   📊 Rating: {rating:.1f}/10 ({votes:,} votes)")
        print(f"   🎭 Genre: {genres}")
        print(f"   🔗 Match Score: {score:.3f}")

    print(f"\n{'='*60}")
    print(f"📌 Hybrid Model: {int(content_weight*100)}% content + {int(rating_weight*100)}% ratings")
    print(f"{'='*60}")

# --- Interactive Loop ---
mode_display = "GPU-ACCELERATED" if GPU_AVAILABLE else "CPU"
print("\n" + "="*60)
print(f" 🎥 {mode_display} MOVIE RECOMMENDER SYSTEM")
print(f" Dataset: {len(df):,} movies")
print(" Features: Content + Ratings + Caching + CUDA")
print(" Commands: 'exit' to quit | 'recache' to rebuild cache")
print("="*60)

while True:
    user_input = input("\n🔍 Enter a movie you like: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Thanks for using the movie recommender!")
        break
    elif user_input.lower() == 'recache':
        print("\n🔄 Rebuilding cache...")
        vectors, vectorizer = train_model()
        save_cache(vectors, vectorizer)
        continue
    
    if user_input:
        hybrid_recommend(user_input)