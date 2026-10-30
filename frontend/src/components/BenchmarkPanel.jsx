import React, { useState, useEffect, useRef, useMemo } from 'react'
import axios from 'axios'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

function BenchmarkPanel({ showToast, fetchStatus, benchmarkSource, onSelectSource, setLoadingOverlay }) {
  const [samples, setSamples] = useState(50)
  const [status, setStatus] = useState({ status: 'idle', progress: 0.0, current: 0, total: 0 })
  const [results, setResults] = useState(null)
  const [isLoadingResults, setIsLoadingResults] = useState(false)
  const [documents, setDocuments] = useState([])
  const [selectedSource, setSelectedSource] = useState(benchmarkSource || null)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)

  // Preview flow state: generate samples for user approval before running full benchmark
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewSamples, setPreviewSamples] = useState([])
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false)

  const sentenceCounterRef = useRef({ val: 0 })
  const answerCounterRef = useRef({ val: 0 })
  const [displaySentenceRate, setDisplaySentenceRate] = useState(0)
  const [displayAnswerRate, setDisplayAnswerRate] = useState(0)
  const pollIntervalRef = useRef(null)

  // Case study filtering, searching, and pagination state
  const [caseFilter, setCaseFilter] = useState('all') // 'all' | 'hallucinated' | 'grounded'
  const [caseSearch, setCaseSearch] = useState('')
  const [casePage, setCasePage] = useState(1)
  const [pageSize, setPageSize] = useState(10) // 5, 10, 20, 50, or 'all'

  // Phase 2 Multi-Signal Fusion state
  const [detectorMetrics, setDetectorMetrics] = useState(null)
  const [isTrainingDetector, setIsTrainingDetector] = useState(false)
  const [activeBenchmarkTab, setActiveBenchmarkTab] = useState('evaluation') // 'evaluation' | 'phase2_signals'

  const activeSource = selectedSource || benchmarkSource || null
  const isSourceScoped = Boolean(activeSource && activeSource.trim())
  const benchmarkTitle = isSourceScoped ? 'Source Benchmark' : 'HaluEval Baseline Benchmark'
  const benchmarkSubtitle = isSourceScoped
    ? 'Click a source card to open its detailed report in a new window.'
    : 'Choose a source card from the benchmark library to view a source-scoped report.'

  // Require an explicit source selection for showing source-scoped results.
  // Only treat results as matching if there is an activeSource and the saved summary points to the same scope.
  const resultsMatchSelectedScope = Boolean(
    results && results.summary &&
    activeSource && activeSource.trim() &&
    (
      (results.summary.benchmark_scope && results.summary.benchmark_scope === activeSource.trim()) ||
      (results.summary.source_filter && results.summary.source_filter === activeSource.trim())
    )
  )

  const caseCounts = useMemo(() => {
    if (!results || !Array.isArray(results.dataset)) return { all: 0, hallucinated: 0, grounded: 0 }
    const total = results.dataset.length
    const hallucinated = results.dataset.filter(d => Boolean(d.has_hallucination)).length
    return {
      all: total,
      hallucinated,
      grounded: total - hallucinated
    }
  }, [results])

  const filteredCases = useMemo(() => {
    if (!results || !Array.isArray(results.dataset)) return []
    let list = results.dataset

    if (caseFilter === 'hallucinated') {
      list = list.filter(item => Boolean(item.has_hallucination))
    } else if (caseFilter === 'grounded') {
      list = list.filter(item => !item.has_hallucination)
    }

    if (caseSearch.trim()) {
      const q = caseSearch.toLowerCase().trim()
      list = list.filter(
        item =>
          (item.question && item.question.toLowerCase().includes(q)) ||
          (item.generated_answer && item.generated_answer.toLowerCase().includes(q)) ||
          (item.reference_knowledge && item.reference_knowledge.toLowerCase().includes(q))
      )
    }

    return list
  }, [results, caseFilter, caseSearch])

  const totalPages = useMemo(() => {
    if (pageSize === 'all') return 1
    const pSize = Number(pageSize) || 10
    return Math.max(1, Math.ceil(filteredCases.length / pSize))
  }, [filteredCases.length, pageSize])

  const paginatedCases = useMemo(() => {
    if (pageSize === 'all') return filteredCases
    const pSize = Number(pageSize) || 10
    const start = (casePage - 1) * pSize
    return filteredCases.slice(start, start + pSize)
  }, [filteredCases, casePage, pageSize])

  useEffect(() => {
    setCasePage(1)
  }, [caseFilter, caseSearch, pageSize])

  const fetchDocuments = async () => {
    try {
      const res = await axios.get('/api/documents')
      setDocuments(res.data || [])
    } catch (e) {
      console.error('Failed to load benchmark library', e)
    }
  }

  const fetchResults = async () => {
    setIsLoadingResults(true)
    try {
      const res = await axios.get('/api/benchmark/results')
      if (res.data.available) {
        setResults(res.data)
      } else {
        setResults(null)
      }
    } catch (e) {
      console.error('Failed to load benchmark results', e)
    } finally {
      setIsLoadingResults(false)
    }
  }

  const fetchDetectorMetrics = async () => {
    try {
      const res = await axios.get('/api/detector/metrics')
      setDetectorMetrics(res.data)
    } catch (e) {
      console.error('Failed to load detector metrics', e)
    }
  }

  const handleTrainDetector = async () => {
    setIsTrainingDetector(true)
    try {
      const res = await axios.post('/api/detector/train')
      setDetectorMetrics(res.data.metrics)
      showToast('Phase 2 fusion classifier trained successfully!', 'success')
    } catch (e) {
      showToast(`Training failed: ${e.response?.data?.detail || e.message}`, 'error')
    } finally {
      setIsTrainingDetector(false)
    }
  }

  useEffect(() => {
    fetchDetectorMetrics()
  }, [])

  const startPollingStatus = () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await axios.get('/api/benchmark/status')
        setStatus(res.data)
        if (typeof res.data.progress === 'number') {
          setLoadingOverlay?.({
            show: true,
            message: 'Running benchmark evaluation...',
            progress: res.data.progress,
          })
        }

        if (res.data.status === 'completed') {
          clearInterval(pollIntervalRef.current)
          showToast('Benchmark run completed! Loading statistics.', 'success')
          fetchResults()
          fetchStatus()
          setLoadingOverlay?.({ show: false, message: '', progress: null })
        } else if (res.data.status === 'failed') {
          clearInterval(pollIntervalRef.current)
          showToast(res.data.error || res.data.message || 'Benchmark execution failed on server.', 'error')
          setLoadingOverlay?.({ show: false, message: '', progress: null })
        }
      } catch (e) {
        console.error('Error polling status:', e)
      }
    }, 2500)
  }

  // New behavior: generate preview samples first, then let user approve to run full benchmark
  const handleRunBenchmark = async () => {
    // Generate preview samples
    setIsGeneratingPreview(true)
    setLoadingOverlay?.({ show: true, message: 'Generating sample preview...', progress: null })

    try {
      const formData = new FormData()
      formData.append('num_samples', samples)
      if (activeSource) {
        formData.append('source_filter', activeSource)
      }
      const res = await axios.post('/api/benchmark/generate_samples', formData)
      if (res.data && Array.isArray(res.data.samples)) {
        setPreviewSamples(res.data.samples)
        setPreviewOpen(true)
      } else {
        showToast('No samples generated for preview.', 'warning')
      }
    } catch (e) {
      showToast(`Failed to generate preview: ${e.response?.data?.detail || e.message}`, 'error')
    } finally {
      setIsGeneratingPreview(false)
      setLoadingOverlay?.({ show: false, message: '', progress: null })
    }
  }

  // Starts the actual benchmark run (used after user approves preview)
  const startBenchmarkRun = async () => {
    setPreviewOpen(false)
    setStatus({ status: 'running', progress: 0, current: 0, total: samples })
    setLoadingOverlay?.({ show: true, message: 'Running benchmark evaluation...', progress: 0 })

    const formData = new FormData()
    formData.append('num_samples', samples)
    formData.append('model_name', 'auto')
    if (activeSource) {
      formData.append('source_filter', activeSource)
    }

    try {
      const res = await axios.post('/api/benchmark/run', formData)
      if (res.data.status === 'started' || res.data.status === 'already_running') {
        showToast(res.data.message, 'warning')
        startPollingStatus()
      } else {
        showToast('Failed to trigger benchmark run.', 'error')
        setStatus({ status: 'idle', progress: 0, current: 0, total: 0 })
      }
    } catch (e) {
      showToast(`Request failed: ${e.response?.data?.detail || e.message}`, 'error')
      setStatus({ status: 'idle', progress: 0, current: 0, total: 0 })
      setLoadingOverlay?.({ show: false, message: '', progress: null })
    }
  }

  const handleSelectSource = (sourceName) => {
    setSelectedSource(sourceName)
    onSelectSource?.(sourceName)
  }

  const handleDownloadReport = () => {
    if (!activeSource?.trim()) {
      showToast('Select a source first to export its benchmark report.', 'warning')
      return
    }

    const params = new URLSearchParams({ source_filter: activeSource.trim() })
    window.open(`/api/benchmark/report?${params.toString()}`, '_blank', 'noopener,noreferrer')
  }

  const handleDeleteReport = async () => {
    if (!activeSource?.trim()) {
      showToast('Select a source first to delete its benchmark report.', 'warning')
      return
    }

    setDeleteConfirmOpen(true)
  }

  const confirmDeleteReport = async () => {
    setDeleteConfirmOpen(false)

    try {
      const params = new URLSearchParams({ source_filter: activeSource.trim() })
      const res = await axios.delete(`/api/benchmark/report?${params.toString()}`)
      showToast(res.data?.message || 'Benchmark report deleted.', 'success')
      setResults(null)
      setStatus({ status: 'idle', progress: 0, current: 0, total: 0 })
      await fetchStatus?.()
      await fetchResults()
    } catch (e) {
      showToast(`Delete failed: ${e.response?.data?.detail || e.message}`, 'error')
    }
  }

  useEffect(() => {
    setSelectedSource(benchmarkSource || null)
  }, [benchmarkSource])

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await axios.get('/api/benchmark/status')
        setStatus(res.data)
        if (res.data.stale_reset) {
          showToast('Cleared stale benchmark state.', 'warning')
        }
        if (res.data.status === 'failed' && res.data.error) {
          showToast(res.data.error, 'error')
        }
        if (res.data.status === 'running') {
          startPollingStatus()
        } else {
          fetchResults()
        }
      } catch (e) {
        console.error(e)
      }
    }

    checkStatus()
    fetchDocuments()

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [])

  useEffect(() => {
    if (!results || !results.summary) return
    setDisplaySentenceRate(Math.round(results.summary.sentence_hallucination_rate))
    setDisplayAnswerRate(Math.round(results.summary.answer_hallucination_rate))
  }, [results])

  const getSentenceChartData = () => {
    if (!results || !results.summary) return []
    return [
      { name: 'Factual', value: results.summary.factual_sentences_count },
      { name: 'Hallucinated', value: results.summary.hallucinated_sentences_count }
    ]
  }

  const getAnswerChartData = () => {
    if (!results || !results.summary) return []
    return [
      { name: 'Grounded', value: results.summary.total_questions - results.summary.hallucinated_answers_count },
      { name: 'Hallucinated', value: results.summary.hallucinated_answers_count }
    ]
  }

  const COLORS_SENTENCE = ['#34c759', '#ff3b30']
  const COLORS_ANSWER = ['#0071e3', '#ff9f0a']

  const libraryCards = useMemo(() => documents, [documents])

  return (
    <div className="flex flex-col gap-5">
      {previewOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-2xl rounded-3xl border border-border-light bg-white p-6 shadow-2xl">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-apple-blue/10 text-apple-blue">
                <i className="fa-solid fa-eye"></i>
              </div>
              <div className="min-w-0">
                <h3 className="text-base font-bold text-text-main">Preview generated QA samples ({previewSamples.length} questions)</h3>
                <p className="mt-1 text-sm leading-relaxed text-text-muted">Review all {previewSamples.length} generated questions and short reference excerpts. Approve to start the full benchmark run.</p>
              </div>
            </div>

            <div className="mt-4 max-h-[56vh] overflow-y-auto">
              {previewSamples.length ? (
                <div className="grid gap-3">
                  {previewSamples.map((s, i) => (
                    <div key={i} className="rounded-lg border border-border-light bg-surface-light p-3">
                      <div className="text-sm font-semibold text-text-main">Q{i + 1}. {s.question}</div>
                      <div className="text-xs text-text-muted mt-1">{(s.reference_knowledge || s.ref || s.knowledge || '').slice(0, 220)}{(s.reference_knowledge || s.ref || s.knowledge || '').length > 220 ? '...' : ''}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-text-muted">No samples available for preview.</div>
              )}
            </div>

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                onClick={() => { setPreviewOpen(false); setPreviewSamples([]) }}
                className="rounded-lg border border-border-light bg-white px-4 py-2 text-xs font-bold text-text-main"
              >
                Cancel
              </button>
              <button
                onClick={startBenchmarkRun}
                disabled={isGeneratingPreview}
                className="rounded-lg bg-text-main px-4 py-2 text-xs font-bold text-white"
              >
                Approve & Run
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteConfirmOpen && activeSource && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-md rounded-3xl border border-border-light bg-white p-6 shadow-2xl">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-apple-red/10 text-apple-red">
                <i className="fa-solid fa-trash"></i>
              </div>
              <div>
                <h3 className="text-base font-bold text-text-main">Delete benchmark report?</h3>
                <p className="mt-1 text-sm leading-relaxed text-text-muted">
                  This will permanently remove the saved benchmark report for <span className="font-semibold text-text-main">{activeSource}</span>.
                </p>
                <p className="mt-2 text-xs text-text-muted">
                  The report download will no longer be available until you run that source benchmark again.
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                onClick={() => setDeleteConfirmOpen(false)}
                className="rounded-lg border border-border-light bg-white px-4 py-2 text-xs font-bold text-text-main"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteReport}
                className="rounded-lg border border-apple-red/20 bg-apple-red/5 px-4 py-2 text-xs font-bold text-apple-red"
              >
                Delete Report
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="apple-panel p-6">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 flex-1">
            <h3 className="text-base font-bold text-text-main flex items-center gap-2">
              <i className="fa-solid fa-vials text-apple-blue"></i> {benchmarkTitle}
            </h3>
            <p className="text-[11px] text-text-muted mt-0.5 font-medium">{benchmarkSubtitle}</p>
            {activeSource && (
              <div className="mt-2 inline-flex max-w-full flex-wrap items-center gap-x-1 rounded-full border border-black/5 bg-white px-3 py-1 text-[11px] font-medium text-text-muted">
                <span>Focused source:</span>
                <span className="min-w-0 break-all font-semibold text-text-main">{activeSource}</span>
              </div>
            )}
          </div>

          <div className="flex w-full flex-wrap items-center gap-3 xl:w-auto xl:justify-end">
            {activeSource && (
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={handleDownloadReport}
                  disabled={status.status === 'running'}
                  className="min-w-0 shrink rounded-lg border border-border-light bg-white px-4 py-2.5 text-xs font-bold text-text-main shadow-sm disabled:opacity-50 flex items-center gap-2"
                  title="Download the complete benchmark report for the selected source"
                >
                  <i className="fa-solid fa-download"></i>
                  Download Report
                </button>
                <button
                  onClick={handleDeleteReport}
                  disabled={status.status === 'running'}
                  className="min-w-0 shrink rounded-lg border border-apple-red/20 bg-apple-red/5 px-4 py-2.5 text-xs font-bold text-apple-red shadow-sm disabled:opacity-50 flex items-center gap-2"
                  title="Delete the saved benchmark report for the selected source"
                >
                  <i className="fa-solid fa-trash"></i>
                  Delete Report
                </button>
              </div>
            )}
            <div className="flex items-center gap-2">
              <label className="text-[9px] font-bold uppercase tracking-wider text-text-muted">Samples</label>
              <select
                value={samples}
                onChange={(e) => setSamples(parseInt(e.target.value))}
                disabled={status.status === 'running'}
                className="bg-surface-light border border-border-light rounded-lg text-xs text-text-main px-3 py-2 outline-none focus:border-apple-blue cursor-pointer disabled:opacity-50 font-medium"
              >
                <option value="10">10 — Quick</option>
                <option value="20">20 — Fast</option>
                <option value="50">50 — Standard</option>
                <option value="100">100 — Detailed</option>
              </select>
            </div>

            <button
              onClick={handleRunBenchmark}
              disabled={status.status === 'running'}
              className="bg-text-main text-white font-bold text-xs px-4 py-2.5 rounded-lg flex items-center gap-2 disabled:opacity-50 shadow-sm"
            >
              <i className="fa-solid fa-play"></i>
              Run Evaluation
            </button>
          </div>
        </div>

        {status.status === 'failed' && status.error && (
          <div className="mt-4 rounded-xl border border-apple-red/20 bg-apple-red/5 px-4 py-3 text-sm text-apple-red">
            {status.error}
          </div>
        )}

        {status.status === 'running' && (
          <div className="mt-5 bg-surface-light border border-border-light p-4 rounded-xl">
            <div className="flex flex-col items-center justify-center gap-4 py-4">
              <div className="flex w-full justify-between text-xs font-semibold">
                <span className="text-text-muted">
                  Evaluating sample {status.current}/{status.total}...
                </span>
                <span className="text-apple-blue font-bold">{Math.round(status.progress)}%</span>
              </div>
              <div className="w-full h-1.5 bg-border-light rounded-full overflow-hidden">
                <div
                  style={{ width: `${status.progress}%` }}
                  className="h-full bg-apple-blue rounded-full"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="apple-panel p-6">
        <div className="mb-4 flex items-end justify-between gap-4 border-b border-border-light pb-4">
          <div>
            <h4 className="text-sm font-bold text-text-main">Benchmark Library</h4>
            <p className="text-[10px] text-text-muted mt-0.5 font-medium">
              Click a source card to view its benchmark result.
            </p>
          </div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">
            {libraryCards.length} source{libraryCards.length === 1 ? '' : 's'}
          </div>
        </div>

        {libraryCards.length ? (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {libraryCards.map((doc) => {
              const isActive = activeSource === doc.source
              const preview = (doc.full_text || '')
                .replace(/\s+/g, ' ')
                .trim()
              const previewShort = preview ? (preview.length > 120 ? preview.slice(0, 117) + '...' : preview) : ''

              return (
                <button
                  key={doc.source}
                  onClick={() => {
                    // mark active visually and open detailed report in a new window (report page will fetch data and render)
                    setSelectedSource(doc.source)
                    onSelectSource?.(doc.source)
                    const params = new URLSearchParams({ source_filter: doc.source })
                    window.open(`/benchmark-report.html?${params.toString()}`, '_blank', 'noopener,noreferrer')
                  }}
                  title={`Open report for ${doc.source}`}
                  className={`text-left rounded-2xl border p-4 min-w-0 transition-colors duration-150 ease-in-out ${isActive
                      ? 'border-apple-blue bg-apple-blue/5 shadow-sm'
                      : 'border-border-light bg-white hover:shadow-sm hover:border-black/5'
                    }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold text-text-main">{doc.source}</div>
                      <div className="mt-1 text-[11px] text-text-muted">{doc.chunks_count ?? 0} chunk{(doc.chunks_count ?? 0) === 1 ? '' : 's'}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-text-muted">View</span>
                      <i className={`fa-solid ${isActive ? 'fa-circle-check text-apple-blue' : 'fa-arrow-up-right-from-square text-text-muted'}`} />
                    </div>
                  </div>

                  {previewShort && (
                    <div className="mt-3 text-[12px] leading-relaxed text-text-muted truncate">
                      {previewShort}
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-border-light bg-surface-light px-5 py-10 text-center text-sm text-text-muted">
            No indexed sources found yet. Upload a document first, then come back to the benchmark library.
          </div>
        )}
      </div>

      {/* Mode Tabs: Phase 1 Evaluation vs. Phase 2 Multi-Signal Fusion */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-border-light pb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveBenchmarkTab('evaluation')}
            className={`rounded-xl px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeBenchmarkTab === 'evaluation'
                ? 'bg-text-main text-white shadow-sm'
                : 'bg-surface-light text-text-muted hover:bg-black/5'
            }`}
          >
            <i className="fa-solid fa-chart-pie mr-1.5" />
            Phase 1: LLM Judge Benchmark
          </button>
          <button
            onClick={() => {
              setActiveBenchmarkTab('phase2_signals')
              fetchDetectorMetrics()
            }}
            className={`rounded-xl px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeBenchmarkTab === 'phase2_signals'
                ? 'bg-apple-blue text-white shadow-sm'
                : 'bg-surface-light text-text-muted hover:bg-black/5'
            }`}
          >
            <i className="fa-solid fa-diagram-project mr-1.5" />
            Phase 2: Multi-Signal Fusion Table
          </button>
        </div>

        {activeBenchmarkTab === 'phase2_signals' && (
          <button
            onClick={handleTrainDetector}
            disabled={isTrainingDetector}
            className="flex items-center gap-2 rounded-xl bg-text-main px-4 py-2 text-xs font-bold text-white hover:bg-black/80 disabled:opacity-60 cursor-pointer"
          >
            <i className={`fa-solid ${isTrainingDetector ? 'fa-spinner fa-spin' : 'fa-brain'}`} />
            {isTrainingDetector ? 'Training Fusion Model...' : 'Re-train Classifier on Labeled Data'}
          </button>
        )}
      </div>

      {/* Phase 2: Multi-Signal Fusion View */}
      {activeBenchmarkTab === 'phase2_signals' && (
        <div className="space-y-6">
          {/* Fusion Model Header Card */}
          <div className="apple-panel p-5 sm:p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-border-light pb-4">
              <div>
                <div className="text-[11px] font-bold uppercase tracking-[0.25em] text-apple-blue">
                  Phase 2 Deliverable: "We Can Catch It"
                </div>
                <h3 className="mt-1 text-lg font-bold text-text-main">
                  Multi-Signal Fusion Classifier (Logistic Regression)
                </h3>
                <p className="mt-1 text-xs text-text-muted">
                  Supervised classifier trained on Phase 1 benchmark data to mathematically combine Token Entropy, Sampling Consistency, and Retrieval Grounding.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="rounded-full bg-apple-green/10 px-3 py-1 text-xs font-bold text-apple-green">
                  ● Status: {detectorMetrics?.status === 'trained' ? 'Trained & Active' : 'Calibrated Baseline'}
                </span>
              </div>
            </div>

            {/* Learned Weights Display */}
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-border-light bg-[#fafafa] p-3 text-center">
                <div className="text-[10px] font-bold uppercase tracking-wider text-text-muted">Entropy Weight (w₁)</div>
                <div className="mt-1 text-base font-extrabold font-mono text-apple-blue">
                  {detectorMetrics?.weights?.entropy ?? 0.25}
                </div>
              </div>
              <div className="rounded-xl border border-border-light bg-[#fafafa] p-3 text-center">
                <div className="text-[10px] font-bold uppercase tracking-wider text-text-muted">Consistency Weight (w₂)</div>
                <div className="mt-1 text-base font-extrabold font-mono text-apple-yellow">
                  {detectorMetrics?.weights?.consistency ?? 0.35}
                </div>
              </div>
              <div className="rounded-xl border border-border-light bg-[#fafafa] p-3 text-center">
                <div className="text-[10px] font-bold uppercase tracking-wider text-text-muted">Grounding Weight (w₃)</div>
                <div className="mt-1 text-base font-extrabold font-mono text-apple-red">
                  {detectorMetrics?.weights?.grounding ?? 0.40}
                </div>
              </div>
              <div className="rounded-xl border border-border-light bg-[#fafafa] p-3 text-center">
                <div className="text-[10px] font-bold uppercase tracking-wider text-text-muted">Model Intercept (b)</div>
                <div className="mt-1 text-base font-extrabold font-mono text-text-main">
                  {detectorMetrics?.weights?.bias ?? -0.10}
                </div>
              </div>
            </div>
          </div>

          {/* VIVA COMPARISON TABLE */}
          <div className="apple-panel p-5 sm:p-6 overflow-hidden">
            <div className="border-b border-border-light pb-4 mb-4">
              <h4 className="text-sm font-bold text-text-main flex items-center gap-2">
                <i className="fa-solid fa-table-list text-apple-blue" />
                Detection Performance Comparison Table (Single Signals vs. Fusion)
              </h4>
              <p className="text-[11px] text-text-muted mt-0.5">
                Evaluated on the benchmark dataset. The fused detector combines independent error indicators, outperforming each individual signal.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-border-light text-[11px] font-bold uppercase tracking-wider text-text-muted">
                    <th className="py-3 px-4">Detection Signal</th>
                    <th className="py-3 px-4 text-center">Accuracy</th>
                    <th className="py-3 px-4 text-center">Precision</th>
                    <th className="py-3 px-4 text-center">Recall</th>
                    <th className="py-3 px-4 text-center font-extrabold text-text-main">F1 Score</th>
                    <th className="py-3 px-4">Key Takeaway</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-light text-xs">
                  {(detectorMetrics?.comparison_table || []).map((row, idx) => {
                    const isFused = row.signal.includes('Fused');
                    return (
                      <tr
                        key={idx}
                        className={`transition-colors ${
                          isFused
                            ? 'bg-apple-blue/5 font-semibold text-text-main'
                            : 'hover:bg-black/[0.01] text-text-dark'
                        }`}
                      >
                        <td className="py-3.5 px-4 font-bold flex items-center gap-2">
                          {isFused && <i className="fa-solid fa-award text-apple-blue" />}
                          {row.signal}
                        </td>
                        <td className="py-3.5 px-4 text-center font-mono">{(row.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-3.5 px-4 text-center font-mono">{(row.precision * 100).toFixed(1)}%</td>
                        <td className="py-3.5 px-4 text-center font-mono">{(row.recall * 100).toFixed(1)}%</td>
                        <td className="py-3.5 px-4 text-center font-mono font-extrabold text-apple-blue">
                          {(row.f1 * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-4 text-[11px] text-text-muted">
                          {idx === 0 && 'Catches model uncertainty, but misses confident hallucinations.'}
                          {idx === 1 && 'Identifies unstable facts, but compute-expensive across samples.'}
                          {idx === 2 && 'Strongest single signal, but sensitive to chunk boundaries.'}
                          {idx === 3 && 'Combines all 3 signals into one optimal decision boundary.'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Confusion Matrix Card */}
          {detectorMetrics?.fused_confusion_matrix && (
            <div className="apple-panel p-5 sm:p-6">
              <h4 className="text-sm font-bold text-text-main mb-3 flex items-center gap-2">
                <i className="fa-solid fa-border-all text-apple-blue" />
                Fused Detector Confusion Matrix
              </h4>
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto sm:grid-cols-4">
                <div className="rounded-xl border border-apple-green/20 bg-apple-green/5 p-3 text-center">
                  <div className="text-[10px] font-bold uppercase text-apple-green">True Positives</div>
                  <div className="text-xl font-extrabold font-mono text-apple-green mt-1">
                    {detectorMetrics.fused_confusion_matrix.true_positive}
                  </div>
                </div>
                <div className="rounded-xl border border-apple-red/20 bg-apple-red/5 p-3 text-center">
                  <div className="text-[10px] font-bold uppercase text-apple-red">False Positives</div>
                  <div className="text-xl font-extrabold font-mono text-apple-red mt-1">
                    {detectorMetrics.fused_confusion_matrix.false_positive}
                  </div>
                </div>
                <div className="rounded-xl border border-apple-yellow/20 bg-apple-yellow/5 p-3 text-center">
                  <div className="text-[10px] font-bold uppercase text-apple-yellow">False Negatives</div>
                  <div className="text-xl font-extrabold font-mono text-apple-yellow mt-1">
                    {detectorMetrics.fused_confusion_matrix.false_negative}
                  </div>
                </div>
                <div className="rounded-xl border border-border-light bg-[#fafafa] p-3 text-center">
                  <div className="text-[10px] font-bold uppercase text-text-muted">True Negatives</div>
                  <div className="text-xl font-extrabold font-mono text-text-main mt-1">
                    {detectorMetrics.fused_confusion_matrix.true_negative}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Phase 1: LLM Judge Benchmark View */}
      {activeBenchmarkTab === 'evaluation' && results && resultsMatchSelectedScope ? (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          <div className="apple-panel p-5 sm:p-6 lg:col-span-4 lg:h-80 flex flex-col items-center justify-between">
            <div className="text-center w-full">
              <h4 className="text-sm font-bold text-text-main">Sentence Hallucination</h4>
              <p className="text-[10px] text-text-muted mt-0.5 font-medium">Factual vs. hallucinated claims</p>
            </div>
            <div className="w-full h-37.5">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={getSentenceChartData()} cx="50%" cy="50%" outerRadius={62} paddingAngle={3} dataKey="value">
                    {getSentenceChartData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS_SENTENCE[index % COLORS_SENTENCE.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e8e8ed', borderRadius: '10px', fontSize: '11px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} itemStyle={{ color: '#1d1d1f' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-3xl font-extrabold font-heading text-apple-red">{displaySentenceRate}%</span>
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider mt-0.5">Hallucination Rate</span>
            </div>
          </div>

          <div className="apple-panel p-5 sm:p-6 lg:col-span-4 lg:h-80 flex flex-col items-center justify-between">
            <div className="text-center w-full">
              <h4 className="text-sm font-bold text-text-main">Answer Hallucination</h4>
              <p className="text-[10px] text-text-muted mt-0.5 font-medium">Answers with ≥1 hallucination</p>
            </div>
            <div className="w-full h-37.5">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={getAnswerChartData()} cx="50%" cy="50%" outerRadius={62} paddingAngle={3} dataKey="value">
                    {getAnswerChartData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS_ANSWER[index % COLORS_ANSWER.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e8e8ed', borderRadius: '10px', fontSize: '11px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} itemStyle={{ color: '#1d1d1f' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-3xl font-extrabold font-heading text-apple-yellow">{displayAnswerRate}%</span>
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider mt-0.5">Hallucinated Answers</span>
            </div>
          </div>

          <div className="apple-panel p-5 sm:p-6 lg:col-span-4 lg:h-80 flex flex-col">
            <div className="mb-4">
              <h4 className="text-sm font-bold text-text-main">Performance Metrics</h4>
              <p className="text-[10px] text-text-muted mt-0.5 font-medium">Statistics from the evaluation run</p>
            </div>
            <div className="flex flex-col justify-between grow">
              {[
                { label: 'Total Questions', value: results.summary.total_questions, color: 'text-text-main' },
                { label: 'Evaluated Sentences', value: results.summary.total_sentences, color: 'text-text-main' },
                { label: 'Factual Sentences', value: results.summary.factual_sentences_count, color: 'text-apple-green' },
                { label: 'Hallucinated Sentences', value: results.summary.hallucinated_sentences_count, color: 'text-apple-red' },
                { label: 'Avg Inference Latency', value: `${results.summary.average_latency}s`, color: 'text-apple-yellow' },
              ].map((row, i) => (
                <div key={i} className={`flex justify-between items-center py-2 ${i < 4 ? 'border-b border-border-light' : ''}`}>
                  <span className="text-[11px] text-text-muted font-medium">{row.label}</span>
                  <span className={`text-xs font-bold font-heading ${row.color}`}>{row.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-12 apple-panel p-5 sm:p-6">
            <div className="border-b border-border-light pb-4 mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h3 className="text-sm font-bold text-text-main flex items-center gap-2">
                  <i className="fa-solid fa-list-check text-apple-blue"></i> Benchmark Case Studies
                  <span className="rounded-full bg-apple-blue/10 text-apple-blue px-2.5 py-0.5 text-xs font-bold">
                    {filteredCases.length} {filteredCases.length === 1 ? 'case' : 'cases'}
                  </span>
                </h3>
                <p className="text-[10px] text-text-muted mt-0.5 font-medium">
                  Sentence-level evaluation — hover over highlighted sentences to view the judge's factual assessment
                </p>
              </div>

              {/* Page size dropdown */}
              <div className="flex items-center gap-2 self-start sm:self-auto">
                <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">Per page:</span>
                <select
                  value={pageSize}
                  onChange={(e) => setPageSize(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                  className="rounded-lg border border-border-light bg-surface-light px-2.5 py-1 text-xs font-semibold text-text-main outline-none focus:border-apple-blue cursor-pointer"
                >
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                  <option value="all">All ({filteredCases.length})</option>
                </select>
              </div>
            </div>

            {/* Filter Pills & Search Bar */}
            <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              {/* Filter Tabs */}
              <div className="flex items-center gap-1.5 rounded-xl border border-border-light bg-base-light p-1">
                <button
                  onClick={() => setCaseFilter('all')}
                  className={`rounded-lg px-3 py-1.5 text-xs font-bold transition-all ${
                    caseFilter === 'all'
                      ? 'bg-white text-text-main shadow-xs'
                      : 'text-text-muted hover:text-text-main'
                  }`}
                >
                  All ({caseCounts.all})
                </button>
                <button
                  onClick={() => setCaseFilter('hallucinated')}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-all ${
                    caseFilter === 'hallucinated'
                      ? 'bg-apple-red text-white shadow-xs'
                      : 'text-apple-red hover:bg-apple-red/10'
                  }`}
                >
                  <i className="fa-solid fa-triangle-exclamation text-[10px]"></i>
                  Hallucinated ({caseCounts.hallucinated})
                </button>
                <button
                  onClick={() => setCaseFilter('grounded')}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-all ${
                    caseFilter === 'grounded'
                      ? 'bg-apple-green text-white shadow-xs'
                      : 'text-apple-green hover:bg-apple-green/10'
                  }`}
                >
                  <i className="fa-solid fa-circle-check text-[10px]"></i>
                  Grounded ({caseCounts.grounded})
                </button>
              </div>

              {/* Search Box */}
              <div className="relative min-w-[240px]">
                <i className="fa-solid fa-magnifying-glass absolute left-3 top-1/2 -translate-y-1/2 text-xs text-text-muted"></i>
                <input
                  type="text"
                  value={caseSearch}
                  onChange={(e) => setCaseSearch(e.target.value)}
                  placeholder="Search questions or answers..."
                  className="w-full rounded-xl border border-border-light bg-surface-light py-1.5 pl-8 pr-8 text-xs font-medium text-text-main outline-none focus:border-apple-blue"
                />
                {caseSearch && (
                  <button
                    onClick={() => setCaseSearch('')}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-text-muted hover:text-text-main"
                  >
                    <i className="fa-solid fa-xmark"></i>
                  </button>
                )}
              </div>
            </div>

            {/* Case Studies Cards List */}
            {paginatedCases.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border-light bg-base-light p-10 text-center">
                <i className="fa-solid fa-filter-circle-xmark text-3xl text-text-muted mb-2 opacity-30"></i>
                <div className="text-sm font-bold text-text-main">No matching case studies</div>
                <p className="mt-1 text-xs text-text-muted">Try adjusting your search terms or filter selection.</p>
                {(caseFilter !== 'all' || caseSearch) && (
                  <button
                    onClick={() => { setCaseFilter('all'); setCaseSearch(''); }}
                    className="mt-3 rounded-lg border border-border-light bg-white px-3 py-1.5 text-xs font-bold text-apple-blue hover:bg-apple-blue/5"
                  >
                    Clear filters
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {paginatedCases.map((ex, idx) => {
                  const originalIndex = ex.sample_index != null ? ex.sample_index : (casePage - 1) * (pageSize === 'all' ? 0 : pageSize) + idx
                  const sTotal = (ex.sentences || []).length
                  const sHallu = (ex.sentences || []).filter(s => s.label === 'hallucinated').length
                  const sFact = sTotal - sHallu

                  return (
                    <div key={idx} className="border border-border-light rounded-2xl overflow-hidden bg-base-light hover:border-black/10 transition-colors shadow-xs">
                      {/* Card Header */}
                      <div className="bg-surface-light px-5 py-3.5 border-b border-border-light flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-apple-blue font-bold uppercase tracking-wider">
                              Case Study #{originalIndex + 1}
                            </span>
                            {ex.category && (
                              <span className="rounded-md bg-border-light px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-text-muted">
                                {ex.category}
                              </span>
                            )}
                            {ex.latency != null && (
                              <span className="text-[10px] text-text-muted font-medium">
                                <i className="fa-regular fa-clock mr-1"></i>{typeof ex.latency === 'number' ? ex.latency.toFixed(2) : ex.latency}s
                              </span>
                            )}
                          </div>
                          <h4 className="text-sm font-bold text-text-main mt-1 leading-snug">
                            "{ex.question}"
                          </h4>
                        </div>

                        {/* Status Badge */}
                        <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
                          <div className={`flex items-center gap-1.5 border px-3 py-1.5 rounded-full text-[10px] font-bold ${
                            ex.has_hallucination
                              ? 'text-apple-red border-apple-red/20 bg-apple-red/5'
                              : 'text-apple-green border-apple-green/20 bg-apple-green/5'
                          }`}>
                            <i className={`fa-solid ${ex.has_hallucination ? 'fa-triangle-exclamation' : 'fa-circle-check'}`}></i>
                            {ex.has_hallucination
                              ? `Hallucination (${sHallu}/${sTotal} sentences)`
                              : `Fully Grounded (${sFact} sentences)`
                            }
                          </div>
                        </div>
                      </div>

                      {/* Card Body */}
                      <div className="p-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
                        {/* Reference Knowledge */}
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <h5 className="text-[10px] font-bold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                              <i className="fa-solid fa-file-signature text-apple-blue"></i> Source Reference
                            </h5>
                            <span className="text-[9px] text-text-muted font-medium">
                              {(ex.reference_knowledge || '').length} chars
                            </span>
                          </div>
                          <div className="bg-white border border-border-light p-4 rounded-xl text-xs leading-relaxed text-text-main h-44 overflow-y-auto font-mono whitespace-pre-wrap select-text">
                            {ex.reference_knowledge || 'No reference knowledge excerpt provided.'}
                          </div>
                        </div>

                        {/* Generated Answer with Judged Sentences */}
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <h5 className="text-[10px] font-bold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
                              <i className="fa-solid fa-comment-dots text-apple-blue"></i> Generated Answer (Judged)
                            </h5>
                            <span className="text-[9px] text-text-muted font-medium">
                              Judge: {ex.judge_method || 'ollama'}
                            </span>
                          </div>
                          <div className="bg-white border border-border-light p-4 rounded-xl text-xs leading-relaxed text-text-main h-44 overflow-y-auto">
                            {Array.isArray(ex.sentences) && ex.sentences.length > 0 ? (
                              ex.sentences.map((s, sIdx) => {
                                const isHallucinated = s.label === 'hallucinated'
                                return (
                                  <span key={sIdx} className={`judged-sentence ${isHallucinated ? 'hallucinated' : 'factual'}`}>
                                    {s.sentence}{' '}
                                    <span className="sentence-tooltip">
                                      <strong className={`${isHallucinated ? 'text-apple-red' : 'text-apple-green'} block font-bold mb-1`}>
                                        {isHallucinated ? 'HALLUCINATION' : 'FACTUAL GROUNDING'}
                                      </strong>
                                      {s.reason || (isHallucinated ? 'Claim not found in reference knowledge' : 'Directly supported by source text')}
                                    </span>
                                  </span>
                                )
                              })
                            ) : (
                              <span className="text-text-muted">{ex.generated_answer || 'No answer available.'}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Pagination Controls */}
            {pageSize !== 'all' && totalPages > 1 && (
              <div className="mt-6 border-t border-border-light pt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="text-xs text-text-muted font-medium">
                  Showing <span className="font-bold text-text-main">{(casePage - 1) * pageSize + 1}</span> to{' '}
                  <span className="font-bold text-text-main">{Math.min(casePage * pageSize, filteredCases.length)}</span> of{' '}
                  <span className="font-bold text-text-main">{filteredCases.length}</span> case studies
                </div>

                <div className="flex items-center gap-1.5 self-center sm:self-auto">
                  <button
                    onClick={() => setCasePage(p => Math.max(1, p - 1))}
                    disabled={casePage === 1}
                    className="rounded-lg border border-border-light bg-white px-3 py-1.5 text-xs font-bold text-text-main disabled:opacity-40 disabled:cursor-not-allowed hover:bg-base-light transition-colors"
                  >
                    <i className="fa-solid fa-chevron-left mr-1"></i> Prev
                  </button>

                  {/* Page pills */}
                  {Array.from({ length: totalPages }, (_, i) => i + 1)
                    .filter(p => p === 1 || p === totalPages || Math.abs(p - casePage) <= 1)
                    .map((p, pIdx, arr) => {
                      const showEllipsisBefore = pIdx > 0 && p - arr[pIdx - 1] > 1
                      return (
                        <React.Fragment key={p}>
                          {showEllipsisBefore && <span className="px-1 text-xs text-text-muted">...</span>}
                          <button
                            onClick={() => setCasePage(p)}
                            className={`h-8 w-8 rounded-lg text-xs font-bold transition-all ${
                              casePage === p
                                ? 'bg-text-main text-white shadow-xs'
                                : 'border border-border-light bg-white text-text-main hover:bg-base-light'
                            }`}
                          >
                            {p}
                          </button>
                        </React.Fragment>
                      )
                    })}

                  <button
                    onClick={() => setCasePage(p => Math.min(totalPages, p + 1))}
                    disabled={casePage === totalPages}
                    className="rounded-lg border border-border-light bg-white px-3 py-1.5 text-xs font-bold text-text-main disabled:opacity-40 disabled:cursor-not-allowed hover:bg-base-light transition-colors"
                  >
                    Next <i className="fa-solid fa-chevron-right ml-1"></i>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : results && activeSource && !resultsMatchSelectedScope ? (
        <div className="apple-panel p-10 py-20 text-center">
          <i className="fa-solid fa-arrows-rotate text-5xl text-text-muted mb-4 opacity-20"></i>
          <h3 className="text-base font-bold text-text-main mb-2">Results belong to a different scope</h3>
          <p className="text-xs text-text-muted max-w-md mx-auto leading-relaxed font-medium">
            The saved benchmark results were generated for <span className="font-semibold text-text-main">{results.summary?.benchmark_scope || 'another scope'}</span>, not the currently selected source.
            Run the benchmark again to generate source-specific results.
          </p>
        </div>
      ) : (
        <div className="apple-panel p-10 py-20 text-center">
          <i className="fa-solid fa-chart-pie text-5xl text-text-muted mb-4 opacity-20"></i>
          <h3 className="text-base font-bold text-text-main mb-2">No Benchmark Data Yet</h3>
          <p className="text-xs text-text-muted max-w-md mx-auto leading-relaxed font-medium">
            {activeSource
              ? 'Run an evaluation grounded in the selected source so the benchmark reflects the currently open document or text.'
              : 'Select a source card or run the baseline HaluEval QA dataset subset.'}
          </p>
        </div>
      )}
    </div>
  )
}

export default BenchmarkPanel
