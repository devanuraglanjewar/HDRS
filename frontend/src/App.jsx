import React, { useEffect, useState } from 'react'
import axios from 'axios'
import Workspace from './components/Workspace'
import ControlCenter from './components/ControlCenter'
import BenchmarkPanel from './components/BenchmarkPanel'
import LoadingOrb from './components/LoadingOrb'

function App() {
  const [activeTab, setActiveTab] = useState('workspace')
  const [benchmarkSource, setBenchmarkSource] = useState(null)
  const [loadingOverlay, setLoadingOverlay] = useState({ show: false, message: '', progress: null })
  const [status, setStatus] = useState({
    vector_store_type: 'Detecting...',
    chromadb_available: false,
    indexed_chunks_count: 0,
    gemini_api_key_configured: false
  })
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' })

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type })
    setTimeout(() => setToast(prev => ({ ...prev, show: false })), 3500)
  }

  const fetchStatus = async () => {
    try {
      const res = await axios.get('/api/status')
      setStatus(res.data)
    } catch (e) {
      console.error('Failed to fetch status', e)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-[#f6f6f4] text-text-main">
      <header className="sticky top-0 z-40 border-b border-black/5 bg-white/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.32em] text-text-muted">
              HDRS
            </div>
            <div className="mt-1 text-sm text-text-muted">
              Phase 1 baseline RAG workspace
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-black/5 bg-white px-2 py-2 shadow-[0_8px_30px_rgba(17,17,26,0.06)]">
            {[
              { key: 'workspace', label: 'Home' },
              { key: 'dashboard', label: 'Control Center' },
              { key: 'benchmark', label: 'Benchmark' }
            ].map((item) => (
              <button
                key={item.key}
                onClick={() => {
                  setActiveTab(item.key)
                  if (item.key !== 'benchmark') setBenchmarkSource(null)
                }}
                className={`rounded-full px-4 py-2 text-xs font-medium ${
                  activeTab === item.key
                    ? 'bg-text-main text-white shadow-md'
                    : 'text-text-muted hover:bg-black/5 hover:text-text-main'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6 lg:px-8 lg:py-8">
        {activeTab === 'workspace' && (
          <Workspace
            showToast={showToast}
            fetchStatus={fetchStatus}
            setLoadingOverlay={setLoadingOverlay}
            onOpenBenchmark={(source) => {
              setBenchmarkSource(source || null)
              setActiveTab('benchmark')
            }}
            onOpenControl={() => setActiveTab('dashboard')}
            onLibraryEmpty={() => setActiveTab('workspace')}
            onBenchmarkReset={() => setActiveTab('workspace')}
          />
        )}
        {activeTab === 'dashboard' && (
          <ControlCenter
            status={status}
            fetchStatus={fetchStatus}
            showToast={showToast}
            setLoadingOverlay={setLoadingOverlay}
            onLibraryEmpty={() => setActiveTab('workspace')}
            onBenchmarkReset={() => setActiveTab('workspace')}
          />
        )}
        {activeTab === 'benchmark' && (
          <BenchmarkPanel
            showToast={showToast}
            fetchStatus={fetchStatus}
            benchmarkSource={benchmarkSource}
            setLoadingOverlay={setLoadingOverlay}
            onSelectSource={setBenchmarkSource}
          />
        )}
      </main>

      {loadingOverlay.show && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-white/92 px-6 backdrop-blur-md">
          <div className="flex w-full max-w-md flex-col items-center gap-5 rounded-[2rem] border border-black/5 bg-white p-8 text-center shadow-[0_30px_100px_rgba(17,17,26,0.18)]">
            <LoadingOrb />
            <div>
              <div className="text-sm font-semibold text-text-main">Loading</div>
              <div className="mt-1 text-sm leading-6 text-text-muted">
                {loadingOverlay.message || 'Please wait while the page finishes the request.'}
              </div>
            </div>
            <div className="w-full">
              <div className="h-2 w-full overflow-hidden rounded-full bg-black/5">
                {typeof loadingOverlay.progress === 'number' ? (
                  <div
                    className="h-full rounded-full bg-text-main transition-all duration-300"
                    style={{ width: `${Math.max(0, Math.min(100, loadingOverlay.progress))}%` }}
                  />
                ) : (
                  <div className="loading-overlay-bar h-full rounded-full bg-text-main" />
                )}
              </div>
              {typeof loadingOverlay.progress === 'number' && (
                <div className="mt-2 text-xs font-semibold text-text-muted">
                  {Math.round(loadingOverlay.progress)}%
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div
        className={`fixed bottom-6 right-6 z-[9999] flex items-center gap-3 rounded-2xl border border-black/5 bg-white px-4 py-3 shadow-[0_18px_50px_rgba(17,17,26,0.12)] ${
          toast.show ? 'translate-y-0 opacity-100' : 'pointer-events-none translate-y-4 opacity-0'
        }`}
      >
        <i
          className={`fa-solid ${
            toast.type === 'success'
              ? 'fa-circle-check text-apple-green'
              : toast.type === 'error'
                ? 'fa-circle-exclamation text-apple-red'
                : 'fa-triangle-exclamation text-amber-500'
          }`}
        />
        <span className="text-sm font-medium text-text-main">{toast.message}</span>
      </div>
    </div>
  )
}

export default App
