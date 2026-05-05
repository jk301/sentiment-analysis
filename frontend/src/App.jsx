import { useState } from 'react'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = async (endpoint) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`http://localhost:8000/api/v1/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })

      // handle non-OK responses
      if (!response.ok) {
        throw new Error('API error')
      }

      const data = await response.json()
      setResult(data)

    } catch (e) {
      setError('Could not reach backend. Is it running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>Sentiment Analysis</h1>

      {/* Input Card */}
      <div className="card">
        <textarea
          rows={5}
          placeholder="Paste customer feedback here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <div className="buttons">
          <button onClick={() => analyze('analyze')} disabled={loading || !text.trim()}>
            LSTM
          </button>
          <button onClick={() => analyze('analyze/pretrained')} disabled={loading || !text.trim()}>
            Pretrained
          </button>
        </div>

        {loading && <p className="status">Analyzing...</p>}
        {error && <p className="error">{error}</p>}
      </div>

      {/* Result Card */}
      {result && (
        <div className="card result-card">
          <h2 className={`label ${result.label?.toLowerCase()}`}>
            {result.label}
          </h2>

          <p className="summary">{result.summary}</p>

          {result.scores && (
            <div className="scores">
              {Object.entries(result.scores).map(([key, value]) => (
                <div key={key} className="score">
                  <span>{key}</span>
                  <div className="bar">
                    <div style={{ width: `${value}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {result.confidence && (
            <p className="confidence">Confidence: {result.confidence}%</p>
          )}
        </div>
      )}
    </div>
  )
}

export default App