import { useState } from 'react'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // calls the backend API and updates the result state
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
      const data = await response.json()
      setResult(data)
    } catch (e) {
      setError('Could not reach backend. Is it running?')
    } finally {
      setLoading(false)
    }
  }

  return (
  <div>
    <h1>Sentiment Analysis</h1>

    {/* textarea where user types feedback */}
    <textarea
      rows={5}
      placeholder="Paste customer feedback here..."
      value={text}
      onChange={(e) => setText(e.target.value)}
    />

    {/* buttons to trigger analysis */}
    <div>
      <button onClick={() => analyze('analyze')} disabled={loading || !text.trim()}>
        Analyze with LSTM
      </button>
      <button onClick={() => analyze('analyze/pretrained')} disabled={loading || !text.trim()}>
        Analyze with Pretrained
      </button>
    </div>

    {/* show loading message while waiting */}
    {loading && <p>Analyzing...</p>}

    {/* show error if backend is unreachable */}
    {error && <p>{error}</p>}

    {/* show result when we get a response */}
    {result && (
      <div>
        <p><strong>Label:</strong> {result.label}</p>
        <p><strong>Summary:</strong> {result.summary}</p>
        {result.scores && (
          <div>
            <p><strong>Scores:</strong></p>
            <p>Positive: {result.scores.Positive}%</p>
            <p>Neutral: {result.scores.Neutral}%</p>
            <p>Negative: {result.scores.Negative}%</p>
          </div>
        )}
        {result.confidence && (
          <p><strong>Confidence:</strong> {result.confidence}%</p>
        )}
      </div>
    )}
  </div>
)
}

export default App