import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [movieInput, setMovieInput] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [recommendations, setRecommendations] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [contentWeight, setContentWeight] = useState(0.7)
  const [numRecommendations, setNumRecommendations] = useState(5)
  const [useGpu, setUseGpu] = useState(true)
  const [stats, setStats] = useState(null)
  const suggestionRef = useRef(null)

  useEffect(() => {
    fetchStats()
  }, [])

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionRef.current && !suggestionRef.current.contains(event.target)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats')
      setStats(response.data)
    } catch (err) {
      console.error('Error fetching stats:', err)
    }
  }

  const searchMovies = async (query) => {
    if (query.length < 2) {
      setSuggestions([])
      return
    }

    try {
      const response = await axios.get(`/api/search?q=${encodeURIComponent(query)}`)
      setSuggestions(response.data)
      setShowSuggestions(true)
    } catch (err) {
      console.error('Error searching movies:', err)
    }
  }

  const handleInputChange = (e) => {
    const value = e.target.value
    setMovieInput(value)
    searchMovies(value)
  }

  const selectMovie = (movie) => {
    setMovieInput(movie.title)
    setShowSuggestions(false)
    setSuggestions([])
  }

  const getRecommendations = async () => {
    if (!movieInput.trim()) {
      setError('Please enter a movie name')
      return
    }

    setLoading(true)
    setError(null)
    setRecommendations(null)

    try {
      const response = await axios.post('/api/recommend', {
        movie: movieInput,
        contentWeight: contentWeight,
        numRecommendations: numRecommendations,
        useGpu: useGpu
      })

      setRecommendations(response.data)
    } catch (err) {
      if (err.response && err.response.data) {
        setError(err.response.data.error)
        if (err.response.data.suggestions) {
          setSuggestions(err.response.data.suggestions)
          setShowSuggestions(true)
        }
      } else {
        setError('An error occurred. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !showSuggestions) {
      getRecommendations()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🎬 Movie Recommender System</h1>
        {stats && (
          <div className="stats">
            <span>📊 {stats.totalMovies?.toLocaleString()} Movies</span>
            {stats.gpuAvailable && <span>🚀 GPU Enabled</span>}
          </div>
        )}
      </header>

      <div className="main-container">
        <div className="search-section">
          <div className="search-box" ref={suggestionRef}>
            <input
              type="text"
              value={movieInput}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Enter a movie name..."
              className="search-input"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div className="suggestions">
                {suggestions.map((movie, idx) => (
                  <div
                    key={idx}
                    className="suggestion-item"
                    onClick={() => selectMovie(movie)}
                  >
                    <span className="suggestion-title">{movie.title}</span>
                    <span className="suggestion-year">({movie.year})</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="controls">
            <div className="control-group">
              <label>
                Content vs Rating Weight: {(contentWeight * 100).toFixed(0)}% / {((1 - contentWeight) * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={contentWeight}
                onChange={(e) => setContentWeight(parseFloat(e.target.value))}
                className="slider"
              />
              <div className="slider-labels">
                <span>More Rating</span>
                <span>More Content</span>
              </div>
            </div>

            <div className="control-group">
              <label>Number of Recommendations: {numRecommendations}</label>
              <input
                type="range"
                min="3"
                max="10"
                step="1"
                value={numRecommendations}
                onChange={(e) => setNumRecommendations(parseInt(e.target.value))}
                className="slider"
              />
            </div>

            {stats?.gpuAvailable && (
              <div className="control-group checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={useGpu}
                    onChange={(e) => setUseGpu(e.target.checked)}
                  />
                  Use GPU Acceleration
                </label>
              </div>
            )}
          </div>

          <button
            onClick={getRecommendations}
            disabled={loading}
            className="recommend-btn"
          >
            {loading ? 'Searching...' : 'Get Recommendations'}
          </button>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>

        {recommendations && (
          <div className="results-section">
            <div className="selected-movie">
              <h2>Selected Movie</h2>
              <div className="movie-card selected">
                <h3>{recommendations.selectedMovie.title}</h3>
                <div className="movie-info">
                  <span>📅 {recommendations.selectedMovie.year}</span>
                  <span>⭐ {recommendations.selectedMovie.rating.toFixed(1)}</span>
                  <span>🗳️ {recommendations.selectedMovie.votes.toLocaleString()} votes</span>
                </div>
              </div>
            </div>

            <div className="recommendations">
              <h2>Recommended Movies</h2>
              <div className="movie-grid">
                {recommendations.recommendations.map((movie, idx) => (
                  <div key={idx} className="movie-card">
                    {movie.upcoming && <span className="upcoming-badge">Upcoming</span>}
                    <h3>{movie.title}</h3>
                    <div className="movie-info">
                      <span>📅 {movie.year}</span>
                      <span>⭐ {movie.rating.toFixed(1)}</span>
                    </div>
                    <div className="movie-genres">{movie.genres}</div>
                    <div className="match-score">
                      Match: {(movie.matchScore * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="search-info">
              <span>⚡ Search completed in {recommendations.searchTime}s</span>
              <span>💻 Mode: {recommendations.mode}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App