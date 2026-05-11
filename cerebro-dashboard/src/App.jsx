import Stream1_FIF from './components/Stream1_FIF'
import Stream2_Epochs from './components/Stream2_Epochs'
import Stream3_AgentLog from './components/Stream3_AgentLog'
import { useCallback, useState } from 'react'

const SUBJECTS = ['A04', 'A05', 'A07', 'A08', 'A09']

export default function App() {
  const [subject, setSubject] = useState('A09')
  const [currentT, setCurrentT] = useState(null)
  const [epochInfo, setEpochInfo] = useState(null)
  const [runActive, setRunActive] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [predictionStatus, setPredictionStatus] = useState('idle')
  const [predictionError, setPredictionError] = useState(null)

  const handleTimeUpdate = useCallback((t) => {
    setCurrentT(t)
  }, [])

  const handleSubjectChange = useCallback((event) => {
    setSubject(event.target.value)
    setCurrentT(null)
    setEpochInfo(null)
    setRunActive(false)
    setSessionId(null)
    setPredictionStatus('idle')
    setPredictionError(null)
  }, [])

  const handleEpochsLoaded = useCallback((info) => {
    setEpochInfo(info)
    setCurrentT(null)
  }, [])

  const handleRunStream = useCallback(() => {
    const stamp = new Date()
      .toISOString()
      .replace(/[-:.TZ]/g, '')
      .slice(0, 14)
    setSessionId(`dashboard_${subject}_${stamp}`)
    setCurrentT(null)
    setEpochInfo(null)
    setPredictionStatus('idle')
    setPredictionError(null)
    setRunActive(true)
  }, [subject])

  const handleStartPrediction = useCallback(async () => {
    if (!sessionId || predictionStatus === 'starting' || predictionStatus === 'started') return
    setPredictionStatus('starting')
    setPredictionError(null)
    try {
      const response = await fetch('/stream/start-prediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject,
          session_id: sessionId,
          interval: 0.5,
          timeout_ms: 120000,
        }),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`)
      }
      setPredictionStatus(data.status === 'already_running' ? 'started' : 'started')
    } catch (err) {
      setPredictionStatus('failed')
      setPredictionError(err.message)
    }
  }, [predictionStatus, sessionId, subject])

  const stream1StartSec = epochInfo
    ? Math.max(0, (epochInfo.anchorEpochStartSec ?? epochInfo.firstEpochStartSec) - 2)
    : null

  return (
    <div style={{
      fontFamily: "'Inter', sans-serif",
      background: '#f8f9fa',
      minHeight: '100vh',
      padding: '20px 24px',
      maxWidth: 1400,
      margin: '0 auto',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 20,
        paddingBottom: 16,
        borderBottom: '2px solid #e5e7eb',
      }}>
        <div>
          <h1 style={{
            fontSize: 22,
            fontWeight: 700,
            color: '#1a1a2e',
            margin: 0,
            fontFamily: 'monospace',
          }}>
            ProjectCerebro
          </h1>
          <p style={{
            fontSize: 12,
            color: '#6b7280',
            margin: '4px 0 0',
          }}>
            Real-Time EEG BCI Dashboard · NYU Tandon School of Engineering
          </p>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 8,
          padding: '6px 14px',
          fontSize: 12,
          color: '#374151',
        }}>
          <span style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: '#22c55e',
            display: 'inline-block',
          }} />
          Subject {subject} · BCI IV-2a · CPU mode
        </div>
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 8,
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            type="button"
            onClick={handleRunStream}
            style={{
              height: 34,
              border: '1px solid #111827',
              borderRadius: 6,
              background: '#111827',
              color: 'white',
              fontSize: 12,
              fontWeight: 700,
              padding: '0 14px',
              cursor: 'pointer',
              fontFamily: 'monospace',
              letterSpacing: 0.5,
              textTransform: 'uppercase',
            }}
          >
            {runActive ? 'Restart Stream' : 'Run Stream'}
          </button>
          {sessionId && (
            <span style={{
              fontSize: 10,
              color: '#6b7280',
              fontFamily: 'monospace',
              background: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              padding: '7px 10px',
            }}>
              session={sessionId}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <label htmlFor="subject-select" style={{
            fontSize: 12,
            color: '#6b7280',
            fontWeight: 600,
          }}>
            Subject
          </label>
          <select
            id="subject-select"
            value={subject}
            onChange={handleSubjectChange}
            style={{
              height: 32,
              border: '1px solid #d1d5db',
              borderRadius: 6,
              background: 'white',
              color: '#374151',
              fontSize: 12,
              padding: '0 10px',
            }}
          >
            {SUBJECTS.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
          {currentT !== null && (
            <span style={{
              fontSize: 11,
              color: '#6b7280',
              fontFamily: 'monospace',
              background: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              padding: '7px 10px',
            }}>
              t = {currentT.toFixed(1)}s
            </span>
          )}
        </div>
      </div>

      {!runActive ? (
        <div style={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '24px',
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          color: '#6b7280',
          fontSize: 12,
        }}>
          Click Run Stream to load epochs, start the raw EEG playback, and attach Stream 3 to a fresh MongoDB session.
        </div>
      ) : stream1StartSec === null ? (
        <div style={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '24px',
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
          color: '#6b7280',
          fontSize: 12,
        }}>
          Loading Stream 2 epochs before starting Stream 1...
        </div>
      ) : (
        <div style={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '20px 24px',
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        }}>
          <Stream1_FIF
            subject={subject}
            startSec={stream1StartSec}
            onTimeUpdate={handleTimeUpdate}
          />
        </div>
      )}

      {runActive && (
        <div style={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '20px 24px',
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        }}>
          <Stream2_Epochs
            subject={subject}
            currentT={currentT}
            onEpochsLoaded={handleEpochsLoaded}
          />
        </div>
      )}

      {runActive && (
        <div style={{
          background: 'white',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: '20px 24px',
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 14,
            paddingBottom: 12,
            borderBottom: '1px solid #f3f4f6',
          }}>
            <div>
              <div style={{
                fontSize: 11,
                fontWeight: 700,
                color: '#1a1a2e',
                fontFamily: 'monospace',
                letterSpacing: 1,
                textTransform: 'uppercase',
              }}>
                Prediction Control
              </div>
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 3 }}>
                Starts Kafka producer and LangGraph consumer for this dashboard session.
              </div>
              {predictionError && (
                <div style={{ fontSize: 11, color: '#dc2626', marginTop: 5 }}>
                  {predictionError}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={handleStartPrediction}
              disabled={!sessionId || predictionStatus === 'starting' || predictionStatus === 'started'}
              style={{
                height: 32,
                border: '1px solid #2563eb',
                borderRadius: 6,
                background: predictionStatus === 'started' ? '#f0fdf4' : '#2563eb',
                color: predictionStatus === 'started' ? '#16a34a' : 'white',
                fontSize: 11,
                fontWeight: 700,
                padding: '0 12px',
                cursor: predictionStatus === 'starting' || predictionStatus === 'started' ? 'default' : 'pointer',
                fontFamily: 'monospace',
                letterSpacing: 0.5,
                textTransform: 'uppercase',
                opacity: predictionStatus === 'starting' ? 0.7 : 1,
              }}
            >
              {predictionStatus === 'starting'
                ? 'Starting...'
                : predictionStatus === 'started'
                  ? 'Prediction Started'
                  : 'Start Prediction'}
            </button>
          </div>
          <Stream3_AgentLog subject={subject} sessionId={sessionId} active={runActive} />
        </div>
      )}
    </div>
  )
}
