import { useCallback, useEffect, useRef, useState } from 'react'
import { useSSE } from '../hooks/useSSE'
import {
  CategoryScale,
  Chart,
  LineController,
  LineElement,
  LinearScale,
  PointElement,
} from 'chart.js'

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
)

const CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ']
const SFREQ = 128
const WINDOW = 10
const MAX_PTS = WINDOW * SFREQ
const VOLTS_TO_MICROVOLTS = 1_000_000
const DISPLAY_LIMIT_UV = 250

const COLORS = {
  FZ: '#9ca3af',
  C3: '#3b82f6',
  CZ: '#22c55e',
  C4: '#ef4444',
  PZ: '#f97316',
}

const LABELS = {
  FZ: 'FZ — frontal',
  C3: 'C3 — left motor cortex',
  CZ: 'CZ — central',
  C4: 'C4 — right motor cortex',
  PZ: 'PZ — parietal',
}

export default function Stream1_FIF({ subject = 'A09', startSec = 0, onTimeUpdate }) {
  const [currentT, setCurrentT] = useState(0)
  const [meta, setMeta] = useState(null)
  const [channels, setChannels] = useState(CHANNELS)
  const [serverError, setServerError] = useState(null)
  const [latestUv, setLatestUv] = useState({})

  const chartRefs = useRef({})
  const canvasRefs = useRef({})
  const buffers = useRef(Object.fromEntries(CHANNELS.map((ch) => [ch, []])))

  useEffect(() => {
    channels.forEach((ch) => {
      const canvas = canvasRefs.current[ch]
      if (!canvas || chartRefs.current[ch]) return

      chartRefs.current[ch] = new Chart(canvas, {
        type: 'line',
        data: {
          labels: new Array(MAX_PTS).fill(''),
          datasets: [{
            data: new Array(MAX_PTS).fill(null),
            borderColor: COLORS[ch] || '#6b7280',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.1,
            fill: false,
          }],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
          },
          scales: {
            x: { display: false },
            y: {
              display: true,
              min: -DISPLAY_LIMIT_UV,
              max: DISPLAY_LIMIT_UV,
              ticks: {
                maxTicksLimit: 3,
                font: { size: 9 },
                color: '#9ca3af',
                callback: (value) => `${value} uV`,
              },
              grid: { color: '#f9fafb' },
            },
          },
        },
      })
    })

    return () => {
      Object.values(chartRefs.current).forEach((chart) => {
        try {
          chart.destroy()
        } catch {
          // Ignore cleanup errors during React StrictMode remounts.
        }
      })
      chartRefs.current = {}
    }
  }, [channels])

  const handleMessage = useCallback((data) => {
    if (data.type === 'error') {
      setServerError(data.error || 'Stream cache unavailable')
      return
    }

    if (data.type === 'meta') {
      setServerError(null)
      setMeta(data)
      setCurrentT(0)
      setLatestUv({})
      if (Array.isArray(data.channels) && data.channels.length > 0) {
        setChannels(data.channels)
        data.channels.forEach((ch) => {
          buffers.current[ch] = []
          if (chartRefs.current[ch]) {
            chartRefs.current[ch].data.datasets[0].data = new Array(MAX_PTS).fill(null)
            chartRefs.current[ch].update('none')
          }
        })
      }
      return
    }

    if (data.type === 'chunk') {
      setCurrentT(data.t)
      if (onTimeUpdate) onTimeUpdate(data.t)
      const nextLatest = {}

      channels.forEach((ch) => {
        const samples = data.channels?.[ch]
        if (!samples || !chartRefs.current[ch]) return

        const buffer = buffers.current[ch]
        for (let i = 0; i < samples.length; i += 1) {
          const microvolts = samples[i] * VOLTS_TO_MICROVOLTS
          const displayValue = Math.max(-DISPLAY_LIMIT_UV, Math.min(DISPLAY_LIMIT_UV, microvolts))
          buffer.push(displayValue)
        }
        nextLatest[ch] = samples[samples.length - 1] * VOLTS_TO_MICROVOLTS
        if (buffer.length > MAX_PTS) {
          buffer.splice(0, buffer.length - MAX_PTS)
        }

        const chart = chartRefs.current[ch]
        chart.data.datasets[0].data = buffer.slice()
        chart.update('none')
      })
      setLatestUv(nextLatest)
    }
  }, [channels])

  const { connected, error } = useSSE(
    `/stream/fif?subject=${encodeURIComponent(subject)}&start_sec=${encodeURIComponent(startSec)}`,
    handleMessage,
  )

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
            Stream 1 — Raw Continuous EEG (.fif)
          </span>
          <span style={{
            fontSize: 11,
            color: '#6b7280',
            marginLeft: 12,
          }}>
            {subject} · {WINDOW}s window · {SFREQ}Hz
            {` · start ${Number(startSec).toFixed(1)}s`}
            {meta && ` · ${(meta.duration / 60).toFixed(1)} min total`}
            {' · display ±'}{DISPLAY_LIMIT_UV}uV
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#6b7280', fontFamily: 'monospace' }}>
            t = {currentT.toFixed(1)}s
          </span>
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

      {(error || serverError) && (
        <div style={{
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 6,
          padding: '6px 12px',
          fontSize: 11,
          color: '#dc2626',
          marginBottom: 8,
        }}>
          {serverError || error}
        </div>
      )}

      {channels.map((ch) => (
        <div key={ch} style={{
          background: 'white',
          border: '1px solid #f3f4f6',
          borderLeft: `3px solid ${COLORS[ch] || '#6b7280'}`,
          borderRadius: '0 6px 6px 0',
          padding: '4px 10px',
          marginBottom: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          <div style={{
            width: 48,
            fontSize: 10,
            fontWeight: 600,
            color: COLORS[ch] || '#6b7280',
            fontFamily: 'monospace',
            flexShrink: 0,
          }}>
            {ch}
          </div>
          <div style={{
            width: 64,
            fontSize: 10,
            color: '#6b7280',
            fontFamily: 'monospace',
            textAlign: 'right',
            flexShrink: 0,
          }}>
            {Number.isFinite(latestUv[ch]) ? `${latestUv[ch].toFixed(1)}uV` : '--'}
          </div>
          <div style={{ flex: 1, height: 55 }}>
            <canvas
              ref={(el) => { canvasRefs.current[ch] = el }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      ))}

      <div style={{
        display: 'flex',
        gap: 16,
        marginTop: 8,
        flexWrap: 'wrap',
      }}>
        {channels.map((ch) => (
          <div key={ch} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 10,
            color: '#6b7280',
          }}>
            <div style={{
              width: 20,
              height: 2,
              background: COLORS[ch] || '#6b7280',
              borderRadius: 1,
            }} />
            {LABELS[ch] || ch}
          </div>
        ))}
      </div>
    </div>
  )
}
