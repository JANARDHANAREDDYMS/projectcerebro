import { useCallback, useEffect, useState } from 'react'
import { useSSE } from '../hooks/useSSE'

const LABEL_COLORS = {
  left: '#3b82f6',
  right: '#ef4444',
  rest: '#6b7280',
}

const LABEL_BG = {
  left: '#eff6ff',
  right: '#fef2f2',
  rest: '#f9fafb',
}

const QUALITY_COLORS = {
  good: '#22c55e',
  noisy: '#f59e0b',
  bad: '#ef4444',
}

const SEVERITY_COLORS = {
  info: '#3b82f6',
  warning: '#f59e0b',
  critical: '#ef4444',
}

const SEVERITY_ICONS = {
  info: 'i',
  warning: '!',
  critical: 'x',
}

const MAX_LOG_ENTRIES = 50
const SHOTS_NEEDED = 50

function ConfidenceBar({ value, color }) {
  const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)))
  return (
    <div style={{
      width: 80,
      height: 6,
      background: '#f3f4f6',
      borderRadius: 3,
      overflow: 'hidden',
      display: 'inline-block',
      verticalAlign: 'middle',
      marginRight: 4,
    }}>
      <div style={{
        width: `${pct}%`,
        height: '100%',
        background: color,
        borderRadius: 3,
        transition: 'width 0.3s ease',
      }} />
    </div>
  )
}

function CalibrationBar({ label, count, color }) {
  const pct = Math.min((count / SHOTS_NEEDED) * 100, 100)
  const done = count >= SHOTS_NEEDED

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 10,
        fontFamily: 'monospace',
        color: '#6b7280',
        marginBottom: 3,
      }}>
        <span style={{ color, fontWeight: 600 }}>{label.toUpperCase()}</span>
        <span>{count}/{SHOTS_NEEDED}{done ? ' done' : ''}</span>
      </div>
      <div style={{
        width: '100%',
        height: 8,
        background: '#f3f4f6',
        borderRadius: 4,
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: done ? '#22c55e' : color,
          borderRadius: 4,
          transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  )
}

export default function Stream3_AgentLog({ subject, sessionId, active = false }) {
  const [log, setLog] = useState([])
  const [stats, setStats] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [connectionMeta, setConnectionMeta] = useState(null)
  const [calStatus, setCalStatus] = useState({
    left: 0,
    right: 0,
    rest: 0,
    status: 'not_started',
  })

  useEffect(() => {
    setLog([])
    setStats(null)
    setAlerts([])
    setConnectionMeta(null)
    setCalStatus({
      left: 0,
      right: 0,
      rest: 0,
      status: 'not_started',
    })
  }, [sessionId])

  const handleMessage = useCallback((data) => {
    if (data.type === 'connected') {
      setConnectionMeta(data)
      return
    }

    if (data.type === 'prediction') {
      setLog((prev) => {
        const entry = {
          id: data.epoch_id,
          time: new Date().toLocaleTimeString(),
          label: data.label_name,
          confidence: data.confidence || 0,
          model: data.model_used,
          quality: data.signal_quality,
          qualityScore: data.quality_score,
          calibration: data.calibration_status,
          n: data.n_predictions,
        }
        return [entry, ...prev].slice(0, MAX_LOG_ENTRIES)
      })

      setStats({
        n_predictions: data.n_predictions,
        n_left: data.n_left,
        n_right: data.n_right,
        n_rest: data.n_rest,
        mean_confidence: data.mean_confidence,
        n_alerts: data.n_alerts,
      })

      setCalStatus((prev) => ({
        left: data.n_left,
        right: data.n_right,
        rest: data.n_rest,
        status: data.calibration_status || prev.status,
      }))

      if (data.alerts && data.alerts.length > 0) {
        setAlerts((prev) => [
          ...data.alerts.map((alert) => ({
            ...alert,
            time: new Date().toLocaleTimeString(),
            epoch_n: data.n_predictions,
          })),
          ...prev,
        ].slice(0, 20))
      }
    }

    if (data.type === 'stats') {
      setStats({
        n_predictions: data.n_predictions,
        n_left: data.n_left,
        n_right: data.n_right,
        n_rest: data.n_rest,
        mean_confidence: data.mean_confidence,
        n_alerts: data.n_alerts,
      })
    }
  }, [])

  const agentStreamUrl = active && sessionId
    ? `/stream/agents?session_id=${encodeURIComponent(sessionId)}`
    : null
  const { connected, error } = useSSE(agentStreamUrl, handleMessage)
  const calibrated = calStatus.status === 'calibrated'
    || (calStatus.left >= SHOTS_NEEDED && calStatus.right >= SHOTS_NEEDED && calStatus.rest >= SHOTS_NEEDED)

  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
      }}>
        <div>
          <span style={{
            fontSize: 12,
            fontWeight: 700,
            color: '#1a1a2e',
            fontFamily: 'monospace',
            letterSpacing: 1,
            textTransform: 'uppercase',
          }}>
            Stream 3 - Agent Output Log
          </span>
          <span style={{ fontSize: 11, color: '#6b7280', marginLeft: 12 }}>
            {subject} · {sessionId || 'no session'} · live predictions · calibration · alerts
          </span>
          {connectionMeta && (
            <span style={{ fontSize: 10, color: '#9ca3af', marginLeft: 12, fontFamily: 'monospace' }}>
              mongo={connectionMeta.database} docs={connectionMeta.initial_count}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444',
          }} />
          <span style={{
            fontSize: 11,
            color: connected ? '#22c55e' : '#ef4444',
            fontWeight: 500,
          }}>
            {connected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </div>

      {error && (
        <div style={{
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 6,
          padding: '6px 12px',
          fontSize: 11,
          color: '#dc2626',
          marginBottom: 8,
        }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 2 }}>
          <div style={{
            fontSize: 10,
            fontWeight: 600,
            color: '#9ca3af',
            fontFamily: 'monospace',
            letterSpacing: 1,
            marginBottom: 6,
            textTransform: 'uppercase',
          }}>
            Live Predictions
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '60px 40px 90px 90px 82px 70px',
            gap: 4,
            fontSize: 9,
            color: '#9ca3af',
            fontFamily: 'monospace',
            fontWeight: 600,
            letterSpacing: 0.5,
            marginBottom: 4,
            paddingBottom: 4,
            borderBottom: '1px solid #f3f4f6',
          }}>
            <span>TIME</span>
            <span>#</span>
            <span>LABEL</span>
            <span>CONF</span>
            <span>MODEL</span>
            <span>QUALITY</span>
          </div>

          <div style={{ height: 280, overflowY: 'auto', fontSize: 11 }}>
            {log.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#9ca3af', fontSize: 12, marginTop: 40 }}>
                Waiting for predictions...
                <div style={{ fontSize: 10, marginTop: 4 }}>
                  Start the Kafka demo to see live agent output
                </div>
              </div>
            ) : log.map((entry, index) => {
              const labelColor = LABEL_COLORS[entry.label] || '#6b7280'
              const isPersonalized = String(entry.model || '').includes('personalized')
              return (
                <div key={`${entry.id}-${index}`} style={{
                  display: 'grid',
                  gridTemplateColumns: '60px 40px 90px 90px 82px 70px',
                  gap: 4,
                  alignItems: 'center',
                  padding: '3px 0',
                  borderBottom: '1px solid #fafafa',
                  background: index === 0 ? '#fafffe' : 'transparent',
                }}>
                  <span style={{ fontSize: 9, color: '#9ca3af', fontFamily: 'monospace' }}>{entry.time}</span>
                  <span style={{ fontSize: 9, color: '#9ca3af', fontFamily: 'monospace' }}>#{entry.n}</span>
                  <span style={{
                    fontSize: 10,
                    fontWeight: 700,
                    color: labelColor,
                    background: LABEL_BG[entry.label] || '#f9fafb',
                    padding: '1px 6px',
                    borderRadius: 4,
                    textAlign: 'center',
                  }}>
                    {entry.label?.toUpperCase() || '?'}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <ConfidenceBar value={entry.confidence} color={labelColor} />
                    <span style={{ fontSize: 10, color: '#374151', fontFamily: 'monospace' }}>
                      {(entry.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span style={{
                    fontSize: 9,
                    color: isPersonalized ? '#22c55e' : '#6b7280',
                    fontFamily: 'monospace',
                    fontWeight: isPersonalized ? 600 : 400,
                  }}>
                    {isPersonalized ? '* pers.' : 'ensemble'}
                  </span>
                  <span style={{
                    fontSize: 9,
                    color: QUALITY_COLORS[entry.quality] || '#6b7280',
                    fontFamily: 'monospace',
                  }}>
                    {entry.quality || '?'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 180 }}>
          <div style={{
            background: '#f9fafb',
            border: '1px solid #f3f4f6',
            borderRadius: 8,
            padding: '10px 12px',
            marginBottom: 12,
          }}>
            <div style={{
              fontSize: 9,
              fontWeight: 700,
              color: '#9ca3af',
              letterSpacing: 1,
              textTransform: 'uppercase',
              marginBottom: 8,
            }}>
              Session Stats
            </div>
            {stats ? (
              <>
                <div style={{
                  fontSize: 18,
                  fontWeight: 700,
                  color: '#1a1a2e',
                  fontFamily: 'monospace',
                  marginBottom: 4,
                }}>
                  {stats.n_predictions}
                  <span style={{ fontSize: 11, fontWeight: 400, color: '#6b7280', marginLeft: 4 }}>
                    predictions
                  </span>
                </div>

                {['left', 'right', 'rest'].map((item) => {
                  const count = stats[`n_${item}`] || 0
                  const pct = stats.n_predictions > 0
                    ? Math.round((count / stats.n_predictions) * 100)
                    : 0
                  return (
                    <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, fontSize: 10 }}>
                      <span style={{ width: 36, color: LABEL_COLORS[item], fontWeight: 600, fontFamily: 'monospace' }}>
                        {item}
                      </span>
                      <div style={{ flex: 1, height: 4, background: '#f3f4f6', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: LABEL_COLORS[item],
                          borderRadius: 2,
                        }} />
                      </div>
                      <span style={{ width: 28, textAlign: 'right', color: '#6b7280', fontFamily: 'monospace', fontSize: 9 }}>
                        {pct}%
                      </span>
                    </div>
                  )
                })}

                <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #f3f4f6', fontSize: 10, color: '#6b7280' }}>
                  <div>Mean conf: <strong>{((stats.mean_confidence || 0) * 100).toFixed(1)}%</strong></div>
                  <div>Alerts: <strong>{stats.n_alerts}</strong></div>
                </div>
              </>
            ) : (
              <div style={{ fontSize: 11, color: '#9ca3af', textAlign: 'center', padding: '12px 0' }}>
                No data yet
              </div>
            )}
          </div>

          <div style={{
            background: '#f9fafb',
            border: '1px solid #f3f4f6',
            borderRadius: 8,
            padding: '10px 12px',
            marginBottom: 12,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <div style={{
                fontSize: 9,
                fontWeight: 700,
                color: '#9ca3af',
                letterSpacing: 1,
                textTransform: 'uppercase',
              }}>
                Calibration
              </div>
              {calibrated && (
                <span style={{
                  fontSize: 9,
                  fontWeight: 700,
                  color: '#22c55e',
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: 4,
                  padding: '1px 6px',
                }}>
                  DONE
                </span>
              )}
            </div>

            <CalibrationBar label="left" count={calStatus.left} color={LABEL_COLORS.left} />
            <CalibrationBar label="right" count={calStatus.right} color={LABEL_COLORS.right} />
            <CalibrationBar label="rest" count={calStatus.rest} color={LABEL_COLORS.rest} />

            <div style={{
              marginTop: 6,
              fontSize: 10,
              color: calibrated ? '#22c55e' : '#6b7280',
              fontWeight: calibrated ? 600 : 400,
              textAlign: 'center',
            }}>
              {calibrated
                ? 'Using personalized model'
                : `${Math.min(calStatus.left, calStatus.right, calStatus.rest)}/${SHOTS_NEEDED} shots`}
            </div>
          </div>

          <div style={{
            background: '#f9fafb',
            border: '1px solid #f3f4f6',
            borderRadius: 8,
            padding: '10px 12px',
          }}>
            <div style={{
              fontSize: 9,
              fontWeight: 700,
              color: '#9ca3af',
              letterSpacing: 1,
              textTransform: 'uppercase',
              marginBottom: 8,
            }}>
              Recent Alerts
            </div>

            {alerts.length === 0 ? (
              <div style={{ fontSize: 10, color: '#9ca3af', textAlign: 'center', padding: '8px 0' }}>
                No alerts
              </div>
            ) : alerts.slice(0, 6).map((alert, index) => (
              <div key={`${alert.message}-${index}`} style={{ display: 'flex', gap: 6, marginBottom: 6, fontSize: 10, alignItems: 'flex-start' }}>
                <span style={{
                  color: SEVERITY_COLORS[alert.severity] || '#6b7280',
                  fontWeight: 700,
                  flexShrink: 0,
                  fontSize: 11,
                }}>
                  {SEVERITY_ICONS[alert.severity] || 'i'}
                </span>
                <div>
                  <div style={{ color: '#374151', lineHeight: 1.3 }}>{alert.message}</div>
                  <div style={{ fontSize: 9, color: '#9ca3af', fontFamily: 'monospace', marginTop: 1 }}>
                    {alert.agent} · #{alert.epoch_n} · {alert.time}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
