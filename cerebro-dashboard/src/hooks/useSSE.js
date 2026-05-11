import { useEffect, useRef, useState } from 'react'

export function useSSE(url, onMessage) {
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)
  const onMessageRef = useRef(onMessage)

  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    if (!url) return undefined

    console.log(`[useSSE] Connecting: ${url}`)
    const source = new EventSource(url)

    source.onopen = () => {
      setConnected(true)
      setError(null)
      console.log(`[useSSE] Connected: ${url}`)
    }

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessageRef.current(data)
      } catch (err) {
        console.error('[useSSE] Parse error:', err)
      }
    }

    source.onerror = () => {
      setConnected(false)
      setError('Connection lost. Retrying...')
    }

    return () => {
      console.log(`[useSSE] Closing: ${url}`)
      source.close()
      setConnected(false)
    }
  }, [url])

  return { connected, error }
}
