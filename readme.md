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
```

Install dependencies:
```bash
pip install pandas scikit-learn numpy torch
```

## 🚀 Getting Started

### 1. Dataset Setup

Place your TMDB movie dataset CSV file at:
```
/mnt/c/Users/abhin/OneDrive/Desktop/Ironfleet/Archives/Content-Based Recommender system/archive/TMDB_movie_dataset_v11.csv
```

Or modify the `CSV_FILE_PATH` variable in the code to point to your dataset location.

**Required CSV columns:**
- `title`: Movie title
- `overview`: Movie description
- `genres`: Movie genres
- `release_date`: Release date
- `original_language`: Language code
- `vote_average`: Rating (0-10)
- `vote_count`: Number of votes

### 2. Run the System

```bash
python code.py
```

### 3. First Run

On first execution, the system will:
1. Load and process the dataset
2. Filter for English-language movies
3. Calculate weighted ratings
4. Train the TF-IDF vectorizer (3000 features)
5. Cache the model for future use

**Initial training time:**
- CPU: ~30-60 seconds
- GPU: ~10-20 seconds

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

| Operation | GPU (CUDA) | CPU |
|-----------|-----------|-----|
| Initial Training | 10-20s | 30-60s |
| Cached Load | 2-5s | 2-5s |
| Similarity Calc | 0.1-0.3s | 2-5s |
| Total Query | <1s | 3-7s |

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

### GPU Not Detected
```
⚠️ CUDA not available, falling back to CPU
```
**Solutions:**
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Check GPU compatibility: `nvidia-smi`

### Movie Not Found
```
❌ Result: 'movie_name' not found in the dataset.
💡 Did you mean one of these?
```
The system suggests close matches. Try:
- Using partial names
- Checking spelling
- Including the year if multiple versions exist

### Cache Mismatch
```
⚠ Cache doesn't match current data, retraining...
```
This happens when the dataset changes. The system automatically retrains.

### Out of Memory (GPU)
Reduce batch size:
```python
batch_size = 1000  # Lower for GPUs with <6GB VRAM
```

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