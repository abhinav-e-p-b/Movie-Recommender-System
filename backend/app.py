from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from datetime import datetime
import numpy as np
from difflib import get_close_matches
import time
import threading
import sys

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
    print("⚠️  PyTorch not found. Falling back to CPU mode...")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
CSV_FILE_PATH = "/mnt/c/Users/abhin/OneDrive/Desktop/Ironfleet/Archives/Content-Based Recommender system/archive/TMDB_movie_dataset_v11.csv"
CACHE_FILE = "movie_recommender_cache_gpu.pkl"

# Global variables for model
df = None
vectors = None
vectorizer = None
recache_lock = threading.Lock()

def cosine_similarity_gpu(query_vector, all_vectors, batch_size=2500):
    """GPU-accelerated cosine similarity"""
    if GPU_AVAILABLE:
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        query_dense = query_vector.toarray().astype(np.float32)
        query_tensor = torch.from_numpy(query_dense).to(device)
        query_norm = torch.norm(query_tensor, dim=1, keepdim=True)
        query_normalized = query_tensor / (query_norm + 1e-10)
        
        n_samples = all_vectors.shape[0]
        similarities = np.zeros(n_samples, dtype=np.float32)
        
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory * 0.7
            optimal_batch = int(free_memory / (3000 * 4 * 2))
            batch_size = min(batch_size, optimal_batch)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_sparse = all_vectors[start_idx:end_idx]
            batch_dense = batch_sparse.toarray().astype(np.float32)
            batch_tensor = torch.from_numpy(batch_dense).to(device)
            
            batch_norm = torch.norm(batch_tensor, dim=1, keepdim=True)
            batch_normalized = batch_tensor / (batch_norm + 1e-10)
            
            batch_similarities = torch.mm(query_normalized, batch_normalized.T)
            similarities[start_idx:end_idx] = batch_similarities.cpu().numpy().flatten()
            
            del batch_dense, batch_tensor, batch_normalized, batch_norm, batch_similarities
            torch.cuda.empty_cache()
        
        del query_tensor, query_normalized
        torch.cuda.empty_cache()
        
        return similarities
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(query_vector, all_vectors).flatten()

def load_data():
    """Load and prepare the movie dataset"""
    global df
    
    print("Loading data from CSV...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        return False
    
    # Ensure all titles are clean strings
    df['title'] = df['title'].fillna('').astype(str)
    
    # Data cleaning
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year.fillna(0).astype(int)
    df = df[df['original_language'] == 'en']
    df = df.reset_index(drop=True)
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('Unknown')
    
    # Handle ratings
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].median()) if 'vote_average' in df.columns else 0
    df['vote_count'] = df['vote_count'].fillna(0) if 'vote_count' in df.columns else 0
    
    # Calculate weighted rating
    C = df['vote_average'].median()
    m = df['vote_count'].quantile(0.60)
    
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        if v + m == 0:
            return C
        return (v / (v + m) * R) + (m / (v + m) * C)
    
    df['weighted_rating'] = df.apply(weighted_rating, axis=1)
    
    # Normalize rating
    rating_range = df['weighted_rating'].max() - df['weighted_rating'].min()
    if rating_range > 0:
        df['rating_score'] = (df['weighted_rating'] - df['weighted_rating'].min()) / rating_range
    else:
        df['rating_score'] = 0.5
    
    # Create tags
    df['tags'] = df['overview'] + " " + df['genres']
    
    print(f"✓ Loaded {len(df):,} movies")
    return True

def train_model():
    """Train the TF-IDF model"""
    global vectorizer
    
    print("Training model...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', 
                                 min_df=2, max_df=0.8, ngram_range=(1, 2))
    vector = vectorizer.fit_transform(df['tags'])
    print(f"✓ Model trained on {len(df):,} movies")
    return vector

def save_cache(vector_matrix, vectorizer_obj):
    """Save cache to disk"""
    # Save the entire processed dataframe for perfect cache matching
    cache_data = {
        'vectors': vector_matrix,
        'vectorizer': vectorizer_obj,
        'df': df.copy(),  # Save the entire processed dataframe
        'timestamp': datetime.now(),
        'gpu_enabled': GPU_AVAILABLE,
        'num_movies': len(df)
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    cache_size = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"✓ Cache saved ({cache_size:.2f} MB)")

def load_cache():
    """Load cached model"""
    global df
    
    if os.path.exists(CACHE_FILE):
        try:
            print("Checking cache...")
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Load the cached dataframe directly
            cached_df = cache_data['df']
            
            # Simple validation: check if number of movies matches
            if len(cached_df) == cache_data['num_movies']:
                df = cached_df  # Use the cached dataframe
                cache_age = datetime.now() - cache_data['timestamp']
                print(f"✓ Loaded cached model ({cache_data['num_movies']:,} movies, {cache_age.days} days old)")
                return cache_data['vectors'], cache_data['vectorizer']
            else:
                print("⚠ Cache validation failed, retraining...")
        except Exception as e:
            print(f"⚠ Error loading cache: {e}")
    return None, None

def rebuild_cache():
    """Rebuild the cache"""
    global vectors, vectorizer
    
    with recache_lock:
        print("\n" + "="*60)
        print("🔄 Starting cache rebuild...")
        print("="*60)
        
        try:
            # Reload data in case CSV changed
            if not load_data():
                print("❌ Failed to reload data")
                return False
            
            # Retrain model
            vectors = train_model()
            
            # Save new cache
            save_cache(vectors, vectorizer)
            
            print("="*60)
            print("✅ Cache rebuild completed successfully!")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Error during cache rebuild: {e}")
            return False

def initialize_model():
    """Initialize the recommendation model"""
    global vectors, vectorizer, df
    
    # Try loading cache first (which includes the processed dataframe)
    vectors, vectorizer = load_cache()
    
    if vectors is None:
        # Cache not found or invalid, so load and process data
        if not load_data():
            return False
        vectors = train_model()
        save_cache(vectors, vectorizer)
    
    print("✓ Model ready!")
    return True

def terminal_listener():
    """Listen for terminal commands in a separate thread"""
    print("\n" + "="*60)
    print("💡 Terminal Commands Available:")
    print("   Type 'recache' or 'r' to rebuild the cache")
    print("   Type 'stats' or 's' to show statistics")
    print("   Type 'help' or 'h' for this help message")
    print("   Type 'quit' or 'q' to shutdown server")
    print("="*60 + "\n")
    
    while True:
        try:
            command = input().strip().lower()
            
            if command in ['recache', 'r']:
                rebuild_cache()
                print("Type another command (or 'help' for options):")
                
            elif command in ['stats', 's']:
                print("\n" + "="*60)
                print("📊 System Statistics:")
                print("="*60)
                print(f"Total Movies: {len(df):,}")
                print(f"GPU Available: {'Yes' if GPU_AVAILABLE else 'No'}")
                if GPU_AVAILABLE:
                    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
                print(f"Cache File: {CACHE_FILE}")
                print(f"Cache Exists: {os.path.exists(CACHE_FILE)}")
                if os.path.exists(CACHE_FILE):
                    cache_size = os.path.getsize(CACHE_FILE) / (1024 * 1024)
                    print(f"Cache Size: {cache_size:.2f} MB")
                print("="*60 + "\n")
                print("Type another command (or 'help' for options):")
                
            elif command in ['help', 'h']:
                print("\n" + "="*60)
                print("💡 Available Commands:")
                print("="*60)
                print("  recache, r  - Rebuild the recommendation cache")
                print("  stats, s    - Show system statistics")
                print("  help, h     - Show this help message")
                print("  quit, q     - Shutdown the server")
                print("="*60 + "\n")
                print("Type a command:")
                
            elif command in ['quit', 'q']:
                print("\n👋 Shutting down server...")
                os._exit(0)
                
            elif command:
                print(f"❌ Unknown command: '{command}'")
                print("Type 'help' for available commands:")
                
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

# Initialize on startup
print("="*60)
print("🎬 Movie Recommender Backend Starting...")
print("="*60)
if not initialize_model():
    print("❌ Failed to initialize model")
    exit(1)

# Start terminal listener thread
listener_thread = threading.Thread(target=terminal_listener, daemon=True)
listener_thread.start()

# --- API Endpoints ---

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'totalMovies': len(df),
        'gpuAvailable': GPU_AVAILABLE,
        'gpuName': torch.cuda.get_device_name(0) if GPU_AVAILABLE else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movie suggestions"""
    query = request.args.get('q', '').strip().lower()
    
    if len(query) < 2:
        return jsonify([])
    
    # Find matching movies
    titles_lower = df['title'].astype(str).str.lower().values
    matches = []
    
    for idx, title in enumerate(titles_lower):
        if query in title:
            matches.append({
                'title': df.iloc[idx]['title'],
                'year': int(df.iloc[idx]['year'])
            })
            if len(matches) >= 10:
                break
    
    return jsonify(matches)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations"""
    data = request.json
    movie_name = data.get('movie', '').strip().lower()
    content_weight = float(data.get('contentWeight', 0.7))
    rating_weight = 1 - content_weight
    num_recommendations = int(data.get('numRecommendations', 5))
    use_gpu = data.get('useGpu', True) and GPU_AVAILABLE
    
    if not movie_name:
        return jsonify({'error': 'No movie name provided'}), 400
    
    # Find movie
    titles_lower = df['title'].str.lower().values
    
    if movie_name not in titles_lower:
        # Find close matches
        close = get_close_matches(movie_name, titles_lower.tolist(), n=5, cutoff=0.6)
        suggestions = []
        for match in close:
            idx = list(titles_lower).index(match)
            suggestions.append({
                'title': df.iloc[idx]['title'],
                'year': int(df.iloc[idx]['year'])
            })
        
        return jsonify({
            'error': f'Movie "{movie_name}" not found',
            'suggestions': suggestions
        }), 404
    
    # Get movie index
    index = list(titles_lower).index(movie_name)
    
    # Calculate similarities
    start_time = time.time()
    movie_vector = vectors[index]
    
    if use_gpu:
        content_scores = cosine_similarity_gpu(movie_vector, vectors)
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        content_scores = cosine_similarity(movie_vector, vectors).flatten()
    
    search_time = time.time() - start_time
    
    # Normalize content scores
    content_range = content_scores.max() - content_scores.min()
    if content_range > 0:
        content_scores_norm = (content_scores - content_scores.min()) / content_range
    else:
        content_scores_norm = content_scores
    
    # Get rating scores
    rating_scores = df['rating_score'].values
    
    # Hybrid scoring
    hybrid_scores = (content_weight * content_scores_norm) + (rating_weight * rating_scores)
    
    # Get top candidates
    similar_movies = sorted(list(enumerate(hybrid_scores)), reverse=True, key=lambda x: x[1])
    top_candidates = similar_movies[1:26]  # Skip the movie itself
    
    # Randomly select from top candidates
    import random
    if len(top_candidates) >= num_recommendations:
        selected_movies = random.sample(top_candidates, num_recommendations)
    else:
        selected_movies = top_candidates
    
    # Prepare response
    input_movie = df.iloc[index]
    today = datetime.now()
    
    recommendations = []
    for movie_idx, score in selected_movies:
        movie = df.iloc[movie_idx]
        release_date = movie['release_date']
        
        upcoming = False
        if pd.notnull(release_date) and release_date > today:
            upcoming = True
        
        recommendations.append({
            'title': movie['title'],
            'year': int(movie['year']),
            'rating': float(movie['vote_average']),
            'votes': int(movie['vote_count']),
            'genres': movie['genres'],
            'matchScore': float(score),
            'upcoming': upcoming
        })
    
    return jsonify({
        'selectedMovie': {
            'title': input_movie['title'],
            'year': int(input_movie['year']),
            'rating': float(input_movie['vote_average']),
            'votes': int(input_movie['vote_count'])
        },
        'recommendations': recommendations,
        'searchTime': round(search_time, 2),
        'totalMovies': len(df),
        'mode': 'GPU (CUDA)' if use_gpu else 'CPU'
    })

@app.route('/api/recache', methods=['POST'])
def recache():
    """Rebuild the cache via API"""
    success = rebuild_cache()
    
    if success:
        return jsonify({'success': True, 'message': 'Cache rebuilt successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to rebuild cache'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Server running on http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)