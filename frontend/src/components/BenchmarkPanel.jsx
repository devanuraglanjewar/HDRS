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
      formData.append('num_samples', Math.min(12, samples))
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
                <h3 className="text-base font-bold text-text-main">Preview generated QA samples</h3>
                <p className="mt-1 text-sm leading-relaxed text-text-muted">Review the generated questions and short reference excerpts. Approve to start the full benchmark run.</p>
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
                  className={`text-left rounded-2xl border p-4 min-w-0 transition-colors duration-150 ease-in-out ${
                    isActive
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

      {results && resultsMatchSelectedScope ? (
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
            <div className="border-b border-border-light pb-4 mb-5">
              <h3 className="text-sm font-bold text-text-main flex items-center gap-2">
                <i className="fa-solid fa-list-check text-apple-blue"></i> Benchmark Case Studies
              </h3>
              <p className="text-[10px] text-text-muted mt-0.5 font-medium">
                Sentence-level evaluation - hover over highlighted text to see the judge's reasoning
              </p>
            </div>

            <div className="flex flex-col gap-5">
              {results.dataset.slice(0, 3).map((ex, idx) => (
                <div key={idx} className="border border-border-light rounded-xl overflow-hidden bg-base-light">
                  <div className="bg-surface-light px-5 py-3.5 border-b border-border-light flex justify-between items-center">
                    <div>
                      <span className="text-[9px] text-apple-blue font-bold uppercase tracking-wider">Case Study #{idx + 1}</span>
                      <h4 className="text-xs font-bold text-text-main mt-0.5">"{ex.question}"</h4>
                    </div>
                    <div className={`flex items-center gap-1.5 border px-3 py-1.5 rounded-full text-[10px] font-bold ${
                      ex.has_hallucination
                        ? 'text-apple-red border-apple-red/20 bg-apple-red/5'
                        : 'text-apple-green border-apple-green/20 bg-apple-green/5'
                    }`}>
                      <i className={`fa-solid ${ex.has_hallucination ? 'fa-triangle-exclamation' : 'fa-circle-check'}`}></i>
                      {ex.has_hallucination ? 'Hallucination Detected' : 'Fully Grounded'}
                    </div>
                  </div>

                  <div className="p-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
                    <div className="flex flex-col gap-2">
                      <h5 className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                        <i className="fa-solid fa-file-signature mr-1 text-apple-blue"></i> Source Reference
                      </h5>
                      <div className="bg-white border border-border-light p-4 rounded-xl text-xs leading-relaxed text-text-main h-44 overflow-y-auto">
                        {ex.reference_knowledge}
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <h5 className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
                        <i className="fa-solid fa-comment-dots mr-1 text-apple-blue"></i> Generated Answer (Judged)
                      </h5>
                      <div className="bg-white border border-border-light p-4 rounded-xl text-xs leading-relaxed text-text-main h-44 overflow-y-auto">
                        {ex.sentences.map((s, sIdx) => {
                          const isHallucinated = s.label === 'hallucinated'
                          return (
                            <span key={sIdx} className={`judged-sentence ${isHallucinated ? 'hallucinated' : 'factual'}`}>
                              {s.sentence}{' '}
                              <span className="sentence-tooltip">
                                <strong className={`${isHallucinated ? 'text-apple-red' : 'text-apple-green'} block font-bold mb-1`}>
                                  {isHallucinated ? 'HALLUCINATION' : 'FACTUAL GROUNDING'}
                                </strong>
                                {s.reason}
                              </span>
                            </span>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
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
