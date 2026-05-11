/**
 * Stream2_Epochs — Synchronized epoch window display.
 *
 * Receives currentT from App.jsx, driven by Stream 1.
 * Loads epoch windows once per subject, then displays the epoch whose original
 * timeline contains currentT. During gaps it shows a flat line.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  CategoryScale,
  Chart,
  LineController,
  LineElement,
  LinearScale,
  PointElement,
} from 'chart.js'

const CURSOR_PLUGIN_ID = 'stream2Cursor'

const cursorPlugin = {
  id: CURSOR_PLUGIN_ID,
  afterDatasetsDraw(chart) {
    const progress = chart.options.plugins?.[CURSOR_PLUGIN_ID]?.progress
    if (progress === null || progress === undefined) return

    const { ctx, chartArea } = chart
    if (!chartArea) return

    const clamped = Math.max(0, Math.min(1, progress))
    const x = chartArea.left + clamped * (chartArea.right - chartArea.left)

    ctx.save()
    ctx.beginPath()
    ctx.moveTo(x, chartArea.top)
    ctx.lineTo(x, chartArea.bottom)
    ctx.lineWidth = 2
    ctx.strokeStyle = '#111827'
    ctx.setLineDash([4, 3])
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(x, chartArea.top + 5, 3, 0, Math.PI * 2)
    ctx.fillStyle = '#111827'
    ctx.fill()
    ctx.restore()
  },
}

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  cursorPlugin,
)

const CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ']
const N_SAMPLES = 512
const FLAT_LINE = new Array(N_SAMPLES).fill(0)
const VOLTS_TO_MICROVOLTS = 1_000_000
const DISPLAY_LIMIT_UV = 15

const COLORS = {
  FZ: '#9ca3af',
  C3: '#3b82f6',
  CZ: '#22c55e',
  C4: '#ef4444',
  PZ: '#f97316',
}

const LABEL_CONFIG = {
  left: {
    color: '#3b82f6',
    bg: '#eff6ff',
    border: '#bfdbfe',
    text: 'LEFT HAND',
  },
  right: {
    color: '#ef4444',
    bg: '#fef2f2',
    border: '#fecaca',
    text: 'RIGHT HAND',
  },
  rest: {
    color: '#6b7280',
    bg: '#f9fafb',
    border: '#e5e7eb',
    text: 'REST / IDLE',
  },
}

function reshapeEpoch(features) {
  const result = {}
  CHANNELS.forEach((ch, i) => {
    result[ch] = Array.from(
      features.slice(i * N_SAMPLES, (i + 1) * N_SAMPLES),
    ).map((value) => {
      const microvolts = value * VOLTS_TO_MICROVOLTS
      return Math.max(-DISPLAY_LIMIT_UV, Math.min(DISPLAY_LIMIT_UV, microvolts))
    })
  })
  return result
}

function findActiveEpoch(epochs, t) {
  return epochs.find(
    (epoch) => t >= epoch.epoch_start_sec && t <= epoch.epoch_end_sec,
  ) || null
}

function findNextEpoch(epochs, t) {
  return epochs.find((epoch) => epoch.epoch_start_sec > t) || null
}

async function readEpochsFromResponse(response, signal) {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    const data = await response.json()
    return data.epochs || []
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const epochs = []
  let buffer = ''
  let expected = null

  while (!signal.aborted) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''

    for (const event of events) {
      const line = event.split('\n').find((item) => item.startsWith('data: '))
      if (!line) continue
      const payload = JSON.parse(line.slice(6))
      if (payload.type === 'error') {
        throw new Error(payload.error || 'Epoch stream unavailable')
      }
      if (payload.type === 'meta') {
        expected = payload.n_epochs
      }
      if (payload.type === 'epoch') {
        epochs.push(payload)
      }
      if (expected !== null && epochs.length >= expected) {
        await reader.cancel()
        return epochs
      }
    }
  }

  return epochs
}

export default function Stream2_Epochs({ currentT, subject, onEpochsLoaded }) {
  const [epochs, setEpochs] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeEpoch, setActiveEpoch] = useState(null)
  const [gapSeconds, setGapSeconds] = useState(null)
  const [error, setError] = useState(null)
  const [timelineT, setTimelineT] = useState(null)
  const [epochProgressSec, setEpochProgressSec] = useState(null)

  const chartRefs = useRef({})
  const canvasRefs = useRef({})

  useEffect(() => {
    if (!subject) return undefined
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setEpochs([])
    setActiveEpoch(null)
    setGapSeconds(null)

    fetch(`/stream/epochs?subject=${encodeURIComponent(subject)}&interval_sec=0.05`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        return readEpochsFromResponse(response, controller.signal)
      })
      .then((loadedEpochs) => {
        const sorted = loadedEpochs
          .slice()
          .sort((a, b) => a.epoch_start_sec - b.epoch_start_sec)
        setEpochs(sorted)
        setLoading(false)
        if (sorted.length > 0 && onEpochsLoaded) {
          const anchorEpoch = sorted[1] || sorted[0]
          onEpochsLoaded({
            firstEpochStartSec: sorted[0].epoch_start_sec,
            firstEpochEndSec: sorted[0].epoch_end_sec,
            anchorEpochStartSec: anchorEpoch.epoch_start_sec,
            anchorEpochEndSec: anchorEpoch.epoch_end_sec,
            anchorEpochIndex: sorted[1] ? 1 : 0,
            nEpochs: sorted.length,
          })
        }
        console.log(`[Stream2] Loaded ${sorted.length} epochs for ${subject}`)
      })
      .catch((err) => {
        if (err.name === 'AbortError') return
        setError(`Failed to load epochs: ${err.message}`)
        setLoading(false)
      })

    return () => controller.abort()
  }, [subject, onEpochsLoaded])

  useEffect(() => {
    CHANNELS.forEach((ch) => {
      const canvas = canvasRefs.current[ch]
      if (!canvas || chartRefs.current[ch]) return

      chartRefs.current[ch] = new Chart(canvas, {
        type: 'line',
        data: {
          labels: new Array(N_SAMPLES).fill(''),
          datasets: [{
            data: FLAT_LINE.slice(),
            borderColor: COLORS[ch],
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
            [CURSOR_PLUGIN_ID]: { progress: null },
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
  }, [])

  const updateCharts = useCallback((channelData) => {
    CHANNELS.forEach((ch) => {
      const chart = chartRefs.current[ch]
      if (!chart) return
      chart.data.datasets[0].data = channelData[ch] || FLAT_LINE
      chart.update('none')
    })
  }, [])

  const updateCursor = useCallback((progress) => {
    CHANNELS.forEach((ch) => {
      const chart = chartRefs.current[ch]
      if (!chart) return
      chart.options.plugins[CURSOR_PLUGIN_ID].progress = progress
      chart.update('none')
    })
  }, [])

  useEffect(() => {
    if (epochs.length === 0 || currentT === null) return

    const timelineTime = currentT
    setTimelineT(timelineTime)

    const active = findActiveEpoch(epochs, timelineTime)
    if (active) {
      setActiveEpoch(active)
      setGapSeconds(null)
      const duration = active.epoch_end_sec - active.epoch_start_sec
      const progressSec = Math.max(0, Math.min(duration, timelineTime - active.epoch_start_sec))
      setEpochProgressSec(progressSec)
      updateCharts(reshapeEpoch(active.features || []))
      updateCursor(duration > 0 ? progressSec / duration : 0)
      return
    }

    setActiveEpoch(null)
    setEpochProgressSec(null)
    const next = findNextEpoch(epochs, timelineTime)
    setGapSeconds(next ? Math.max(0, Math.round(next.epoch_start_sec - timelineTime)) : null)
    updateCharts(Object.fromEntries(CHANNELS.map((ch) => [ch, FLAT_LINE])))
    updateCursor(null)
  }, [currentT, epochs, updateCharts, updateCursor])

  const label = activeEpoch
    ? LABEL_CONFIG[activeEpoch.label_name] || LABEL_CONFIG.rest
    : null
  const isGap = !activeEpoch

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
            Stream 2 — Synchronized Epoch Windows
          </span>
          <span style={{
            fontSize: 11,
            color: '#6b7280',
            marginLeft: 12,
          }}>
            {subject} · 4s window · 128Hz · {epochs.length} epochs
            {timelineT !== null && ` · timeline ${timelineT.toFixed(1)}s`}
            {` · display +/-${DISPLAY_LIMIT_UV}uV`}
          </span>
        </div>

        {loading ? (
          <span style={{
            fontSize: 11,
            color: '#6b7280',
            background: '#f3f4f6',
            padding: '3px 10px',
            borderRadius: 12,
          }}>
            Loading epochs...
          </span>
        ) : activeEpoch ? (
          <span style={{
            fontSize: 11,
            fontWeight: 700,
            color: label.color,
            background: label.bg,
            border: `1px solid ${label.border}`,
            padding: '3px 12px',
            borderRadius: 12,
          }}>
            {label.text}
          </span>
        ) : (
          <span style={{
            fontSize: 11,
            color: '#9ca3af',
            background: '#f9fafb',
            border: '1px solid #e5e7eb',
            padding: '3px 12px',
            borderRadius: 12,
          }}>
            {gapSeconds !== null
              ? `Gap - ${gapSeconds}s until next epoch`
              : 'Waiting for stream...'}
          </span>
        )}
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

      {CHANNELS.map((ch) => (
        <div key={ch} style={{
          background: isGap ? '#fafafa' : 'white',
          border: '1px solid #f3f4f6',
          borderLeft: `3px solid ${isGap ? '#e5e7eb' : COLORS[ch]}`,
          borderRadius: '0 6px 6px 0',
          padding: '4px 10px',
          marginBottom: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          opacity: isGap ? 0.5 : 1,
          transition: 'opacity 0.3s, border-left-color 0.3s',
        }}>
          <div style={{
            width: 48,
            fontSize: 10,
            fontWeight: 600,
            color: isGap ? '#9ca3af' : COLORS[ch],
            fontFamily: 'monospace',
            flexShrink: 0,
          }}>
            {ch}
          </div>
          <div style={{ flex: 1, height: 55 }}>
            <canvas
              ref={(el) => { canvasRefs.current[ch] = el }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      ))}

      {activeEpoch && (
        <div style={{
          display: 'flex',
          gap: 16,
          marginTop: 6,
          fontSize: 10,
          color: '#9ca3af',
          fontFamily: 'monospace',
        }}>
          <span>start={activeEpoch.epoch_start_sec.toFixed(1)}s</span>
          <span>end={activeEpoch.epoch_end_sec.toFixed(1)}s</span>
          <span>duration=4.0s</span>
          <span>cursor={epochProgressSec !== null ? epochProgressSec.toFixed(1) : '--'}s</span>
          <span style={{ color: label?.color, fontWeight: 600 }}>
            {activeEpoch.label_name?.toUpperCase()}
          </span>
        </div>
      )}
    </div>
  )
}
