# 🎬 GPU-Accelerated Movie Recommender System

A high-performance, hybrid content-based movie recommendation system with GPU acceleration support, featuring intelligent caching and real-time filtering across a comprehensive movie database.

## ✨ Features

- **🚀 GPU Acceleration**: CUDA-enabled similarity calculations for lightning-fast recommendations
- **🎯 Hybrid Scoring**: Combines content similarity (70%) with weighted ratings (30%)
- **💾 Smart Caching**: Persistent model storage for instant startup
- **📊 Advanced Rating System**: IMDB-style weighted ratings with vote count consideration
- **🔜 Upcoming Movies**: Includes unreleased films with clear indicators
- **🎲 Variety Mode**: Randomized selection from top 25 matches for diverse recommendations
- **⚡ Memory Optimized**: Efficient batch processing for systems with 6GB+ VRAM

## 🛠️ Technical Stack

- **Machine Learning**: scikit-learn (TF-IDF vectorization)
- **GPU Computing**: PyTorch with CUDA support
- **Data Processing**: pandas, NumPy
- **Storage**: pickle for model persistence

## 📋 Requirements

```
pandas
scikit-learn
numpy
torch (with CUDA support for GPU acceleration)
kagglehub (for automatic dataset download)
```

Install all dependencies:
```bash
pip install pandas scikit-learn numpy torch kagglehub
```

For GPU support, install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Getting Started

### 1. Dataset Setup

#### Option A: Download from Kaggle (Recommended)

Install kagglehub and download the dataset automatically:

```bash
pip install kagglehub
```

Then run this Python script or add it to the beginning of your code:

```python
import kagglehub

# Download latest version of TMDB dataset
path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
print("Path to dataset files:", path)
```

**Dataset Information:**
- **Source**: The Movie Database (TMDB)
- **Size**: ~1,000,000 movies (930K+ movies)
- **Year**: 2023 version
- **Content**: Comprehensive movie metadata including titles, ratings, release dates, revenue, genres, cast, crew, and more

#### Option B: Use Existing Dataset

If you already have the dataset, update the `CSV_FILE_PATH` variable in the code:

```python
CSV_FILE_PATH = "path/to/your/TMDB_movie_dataset_v11.csv"
```

**Required CSV columns:**
- `title`: Movie title
- `overview`: Movie description/synopsis
- `genres`: Movie genres
- `release_date`: Release date
- `original_language`: Language code (ISO 639-1)
- `vote_average`: Average rating (0-10 scale)
- `vote_count`: Number of user votes

### 2. Run the System

```bash
python code.py
```

### 3. First Run

On first execution, the system will:
1. Download the TMDB dataset (if using kagglehub) - ~500MB download
2. Load and process ~1,000,000 movies
3. Filter for English-language movies
4. Calculate weighted ratings using IMDB formula
5. Train the TF-IDF vectorizer (3000 features, bigrams)
6. Create and cache the similarity model

**Initial training time:**
- CPU: ~1-2 minutes (for ~85K English movies after filtering)
- GPU: ~30-45 seconds

**Dataset Statistics:**
- Total movies: ~1,000,000
- English movies: ~85,000-100,000 (after filtering)
- Features: Overview text + Genres
- Vector dimensions: 3000 TF-IDF features

### 4. Subsequent Runs

The cached model loads in **~2-5 seconds**, providing instant recommendations.

## 💡 Usage

### Interactive Mode

```
🔍 Enter a movie you like: inception
```

The system will display:
- 5 personalized recommendations
- Ratings and vote counts
- Genres
- Match scores
- Upcoming movie indicators

### Special Commands

- **`exit`** / **`quit`** / **`q`**: Exit the program
- **`recache`**: Rebuild the model cache (use after dataset updates)

## 🎯 How It Works

### Hybrid Scoring Algorithm

```
Hybrid Score = (0.7 × Content Similarity) + (0.3 × Normalized Rating)
```

**Content Similarity (70%)**
- TF-IDF vectorization of movie overviews and genres
- Cosine similarity calculation
- 3000 features with bigrams (1-2 word phrases)

**Rating Score (30%)**
- IMDB weighted rating formula:
  ```
  WR = (v/(v+m) × R) + (m/(v+m) × C)
  ```
  - `v`: vote count
  - `m`: minimum votes threshold (60th percentile)
  - `R`: average rating
  - `C`: mean rating across all movies

### GPU Acceleration

**Memory-Efficient Batch Processing:**
- Processes movies in batches of 2,500
- Uses only 70% of available VRAM
- Batch size: ~80MB per iteration (safe for 6GB VRAM)
- Automatic CPU fallback if CUDA unavailable

**Performance:**
- GPU: ~0.1-0.3 seconds for 100,000+ movies
- CPU: ~2-5 seconds for 100,000+ movies

## 📊 Performance Metrics

| Operation | GPU (CUDA) | CPU | Dataset Size |
|-----------|-----------|-----|--------------|
| Dataset Download | ~2-5 min | ~2-5 min | 500MB |
| Initial Training | 30-45s | 1-2 min | ~85K movies |
| Cached Load | 2-5s | 2-5s | - |
| Similarity Calc | 0.1-0.3s | 2-5s | Per query |
| Total Query | <1s | 3-7s | - |

**System Requirements:**
- RAM: 4GB minimum, 8GB recommended
- Storage: 1GB for dataset + cache
- GPU: 6GB VRAM recommended for full GPU acceleration

## 🎨 Example Output

```
⏳ Searching through 85,432 movies using GPU (CUDA)...
⚡ Similarity calculated in 0.18 seconds

============================================================
🎬 Recommendations for 'Inception' (2010)
   Rating: 8.4/10 (31,456 votes)
   Searched: 85,432 movies
============================================================

1. Interstellar (2014)
   📊 Rating: 8.6/10 (29,834 votes)
   🎭 Genre: Science Fiction, Drama, Adventure
   🔗 Match Score: 0.847

2. The Prestige (2006)
   📊 Rating: 8.2/10 (24,109 votes)
   🎭 Genre: Drama, Mystery, Thriller
   🔗 Match Score: 0.821

...

============================================================
📌 Hybrid Model: 70% content + 30% ratings
============================================================
```

## 🔧 Configuration

Customize the recommendation behavior in the code:

```python
# Hybrid weights
content_weight = 0.7    # Content similarity importance
rating_weight = 0.3     # Rating quality importance

# Number of recommendations
num_recommendations = 5  # Default: 5

# TF-IDF parameters
max_features = 3000     # Vocabulary size
min_df = 2              # Minimum document frequency
max_df = 0.8            # Maximum document frequency

# GPU batch size
batch_size = 2500       # Adjust based on VRAM
```

## 🐛 Troubleshooting

### Dataset Download Issues
```
Error: Failed to download dataset
```
**Solutions:**
- Ensure you have a Kaggle account
- Configure Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
- Check internet connection
- Verify dataset availability: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

### GPU Not Detected
```
⚠️ CUDA not available, falling back to CPU
```
**Solutions:**
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Check GPU compatibility: `nvidia-smi`
- Verify CUDA installation: `torch.cuda.is_available()` in Python

### Movie Not Found
```
❌ Result: 'movie_name' not found in the dataset.
💡 Did you mean one of these?
```
The system suggests close matches using fuzzy matching. Try:
- Using partial names (e.g., "dark knight" instead of "The Dark Knight")
- Checking spelling and punctuation
- Including the year if multiple versions exist (handled automatically)
- Searching for the original title if it's a foreign film

### Cache Mismatch
```
⚠ Cache doesn't match current data, retraining...
```
This happens when:
- Dataset file is updated
- CSV structure changes
- Manual cache corruption

The system automatically retrains. Use `recache` command to force rebuild.

### Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```
Reduce batch size in the code:
```python
batch_size = 1000  # Lower for GPUs with <6GB VRAM
batch_size = 500   # For 4GB VRAM
```

### Large Dataset Performance
For the full 1M movie dataset, consider:
- Increasing `min_df` parameter to filter rare words
- Reducing `max_features` from 3000 to 2000
- Using more aggressive filtering (higher vote count threshold)

## 📈 Future Enhancements

- [ ] User-based collaborative filtering
- [ ] Multi-GPU support for massive datasets
- [ ] Web interface (Flask/FastAPI)
- [ ] Personalized user profiles
- [ ] A/B testing different hybrid weights
- [ ] Export recommendations to JSON/CSV
- [ ] Integration with streaming service APIs

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional metadata features (directors, actors, keywords)
- Alternative similarity metrics
- Real-time dataset updates
- Performance optimizations
- UI/UX enhancements

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for movie enthusiasts**