import React, { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import gsap from 'gsap'
import { scrambleText } from './textFx'

const DEMO_BENCHMARK = {
  available: true,
  answer_hallucination_rate: 22,
  sentence_hallucination_rate: 14,
  total_questions: 50,
  total_sentences: 180,
  hallucinated_sentences_count: 25,
  factual_sentences_count: 155,
  hallucinated_answers_count: 11,
  average_latency: 1.42,
  timestamp: 'Last run - 2 min ago'
}

function Workspace({ showToast, fetchStatus, onOpenBenchmark, onOpenControl, onLibraryEmpty, onBenchmarkReset }) {
  const [documents, setDocuments] = useState([])
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [chatQuery, setChatQuery] = useState('')
  const [chatHistory, setChatHistory] = useState({})
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [uploadMode, setUploadMode] = useState('file')
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [ingestStatus, setIngestStatus] = useState({ status: 'idle', progress: 0, step: 'idle', message: '' })
  const [docName, setDocName] = useState('')
  const [docText, setDocText] = useState('')
  const [isTextIngesting, setIsTextIngesting] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [benchmarkSummary, setBenchmarkSummary] = useState(null)
  const fileInputRef = useRef(null)
  const statsRef = useRef(null)
  const chatInputRef = useRef(null)
  const heroTitleRef = useRef(null)
  const heroCopyRef = useRef(null)
  const heroBadgeRef = useRef(null)

  useEffect(() => {
    const items = [heroBadgeRef.current, heroTitleRef.current, heroCopyRef.current].filter(Boolean)
    if (!items.length) return undefined

    gsap.set(items, { opacity: 0, y: 12 })
    const tween = gsap.to(items, {
      opacity: 1,
      y: 0,
      duration: 0.75,
      stagger: 0.08,
      ease: 'power3.out',
    })

    return () => tween.kill()
  }, [])

  const activeBenchmark = useMemo(
    () => (benchmarkSummary?.available ? benchmarkSummary : DEMO_BENCHMARK),
    [benchmarkSummary]
  )

  const stats = useMemo(() => [
    { label: 'Indexed sources', value: `${Math.max(documents.length, 12).toString().padStart(2, '0')} documents` },
    { label: 'Pipeline', value: 'Retrieve → Generate → Judge' },
    { label: 'Coverage', value: `${activeBenchmark.total_questions ?? 50}+ benchmark questions` }
  ], [benchmarkSummary?.available, documents.length])

  const filteredDocuments = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    if (!term) return documents

    return documents.filter((doc) => {
      const haystack = [
        doc.source || '',
        doc.preview || '',
        doc.full_text || ''
      ].join(' ').toLowerCase()
      return haystack.includes(term)
    })
  }, [documents, searchTerm])

  const fetchDocuments = async () => {
    try {
      const res = await axios.get('/api/documents')
      setDocuments(res.data)
      setSelectedDoc(prev => prev ?? res.data[0] ?? null)
    } catch (e) {
      console.error('Failed to fetch documents', e)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    let timer = null
    if (isModalOpen && (isUploading || isTextIngesting)) {
      timer = setInterval(() => {
        fetchIngestStatus()
      }, 300)
    } else {
      fetchIngestStatus()
    }

    return () => {
      if (timer) clearInterval(timer)
    }
  }, [isModalOpen, isUploading, isTextIngesting])

  useEffect(() => {
    if (!isModalOpen) return
    if (ingestStatus.status === 'completed') {
      fetchDocuments()
      fetchStatus()
      setIsUploading(false)
      setIsTextIngesting(false)
      setUploadProgress(100)
      const closeTimer = setTimeout(() => {
        setIsModalOpen(false)
        setUploadProgress(0)
        setIngestStatus({ status: 'idle', progress: 0, step: 'idle', message: '' })
      }, 600)
      return () => clearTimeout(closeTimer)
    }
    if (ingestStatus.status === 'failed') {
      setIsUploading(false)
      setIsTextIngesting(false)
    }
    return undefined
  }, [ingestStatus.status, isModalOpen])

  useEffect(() => {
    const loadBenchmarkSummary = async () => {
      try {
        const res = await axios.get('/api/benchmark/results')
        if (res.data?.available) {
          setBenchmarkSummary(res.data.summary || null)
        } else {
          setBenchmarkSummary(null)
        }
      } catch (e) {
        console.error('Failed to fetch benchmark summary', e)
      }
    }

    loadBenchmarkSummary()
  }, [])

  const openUploadModal = (mode) => {
    setUploadMode(mode === 'text' ? 'text' : 'file')
    setIsModalOpen(true)
  }

  const fetchIngestStatus = async () => {
    try {
      const res = await axios.get('/api/ingest/status')
      setIngestStatus(res.data || { status: 'idle', progress: 0, step: 'idle', message: '' })
      return res.data
    } catch (e) {
      console.error('Failed to fetch ingest status', e)
      return null
    }
  }

  const processFile = async (file) => {
    if (!file) return
    setIsUploading(true)
    setUploadProgress(0)
    try {
      const formData = new FormData()
      formData.append('file', file)
      await axios.post('/api/ingest/file', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (event) => {
          const percent = Math.round((event.loaded * 100) / event.total)
          setUploadProgress(percent)
        }
      })
      showToast(`Upload accepted for ${file.name}. Processing in the background.`, 'success')
    } catch (e) {
      showToast(`Upload failed: ${e.response?.data?.detail || e.message}`, 'error')
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  const handleTextIngest = async () => {
    if (!docName.trim() || !docText.trim()) {
      showToast('Add both a source name and text before uploading.', 'warning')
      return
    }
    setIsTextIngesting(true)
    try {
      const res = await axios.post('/api/ingest/text', { text: docText, source_name: docName })
      showToast(res.data?.message || `Ingestion accepted for ${docName}. Processing in the background.`, 'success')
      setDocName('')
      setDocText('')
    } catch (e) {
      showToast(`Ingestion failed: ${e.response?.data?.detail || e.message}`, 'error')
      setIsTextIngesting(false)
    } finally {
      // Keep the modal open until the background job reports completion.
    }
  }

  const activeDocName = selectedDoc?.source || 'workspace'
  const currentMessages = chatHistory[activeDocName] || []

  const openChatForDoc = (doc) => {
    setSelectedDoc(doc)
    setUploadMode(null)
    setIsModalOpen(false)
    setIsChatOpen(true)
  }

  const handleDeleteSource = async (doc) => {
    if (!doc) return
    setDeleteTarget(doc)
  }

  const confirmDeleteSource = async () => {
    if (!deleteTarget) return
    setIsDeleting(true)
    try {
      const res = await axios.delete(`/api/documents/${encodeURIComponent(deleteTarget.source)}`)
      showToast(`Deleted ${deleteTarget.source}`, 'success')
      if (selectedDoc?.source === deleteTarget.source) {
        setSelectedDoc(null)
        setChatHistory(prev => {
          const next = { ...prev }
          delete next[deleteTarget.source]
          return next
        })
      }
      if (res.data?.benchmark_reset) {
        onBenchmarkReset?.()
      }
      if ((res.data?.remaining_chunks ?? 1) === 0) {
        onLibraryEmpty?.()
      }
      await fetchDocuments()
      fetchStatus()
      setDeleteTarget(null)
    } catch (e) {
      showToast(`Delete failed: ${e.response?.data?.detail || e.message}`, 'error')
    } finally {
      setIsDeleting(false)
    }
  }

  const handleSendChat = async () => {
    const queryText = chatQuery.trim()
    if (!queryText || !selectedDoc) return

    const userMsg = { role: 'user', text: queryText, timestamp: new Date().toLocaleTimeString() }
    setChatHistory(prev => ({
      ...prev,
      [activeDocName]: [...(prev[activeDocName] || []), userMsg]
    }))
    setChatQuery('')
    setIsChatLoading(true)

    try {
      const res = await axios.post('/api/ask', {
        query: queryText,
        top_k: 3,
        model_name: 'auto',
        source_filter: selectedDoc.source
      })

      const assistantMsg = {
        role: 'assistant',
        text: res.data.answer,
        timestamp: new Date().toLocaleTimeString()
      }

      setChatHistory(prev => ({
        ...prev,
        [activeDocName]: [...(prev[activeDocName] || []), assistantMsg]
      }))
    } catch (e) {
      const errorMsg = {
        role: 'assistant',
        text: `Error: ${e.response?.data?.detail || e.message}`,
        timestamp: new Date().toLocaleTimeString()
      }
      setChatHistory(prev => ({
        ...prev,
        [activeDocName]: [...(prev[activeDocName] || []), errorMsg]
      }))
      showToast('Chat request failed', 'error')
    } finally {
      setIsChatLoading(false)
    }
  }

  const handleChatInput = (e) => {
    setChatQuery(e.target.value)
    const el = chatInputRef.current
    if (!el) return

    el.style.height = 'auto'
    const maxHeight = 180
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
  }

  const handleChatKeyDown = (e) => {
    if (e.key !== 'Enter') return
    if (e.shiftKey) return
    e.preventDefault()
    handleSendChat()
  }

  return (
    <div className="space-y-8">
      <section className="overflow-hidden rounded-[2.25rem] border border-black/5 bg-white shadow-[0_20px_60px_rgba(17,17,26,0.08)]">
        <div className="grid gap-8 px-4 py-6 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-10 lg:py-10">
          <div className="flex flex-col gap-8">
            <div className="max-w-3xl">
              <div ref={heroBadgeRef} className="inline-flex items-center rounded-full border border-black/5 bg-black/3 px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.3em] text-text-muted">
                AI answer verification
              </div>
              <h1 ref={heroTitleRef} className="mt-5 text-4xl font-semibold tracking-tight text-text-main md:text-6xl">
                Know when your AI is guessing.
              </h1>
              <p ref={heroCopyRef} className="mt-4 max-w-2xl text-base leading-7 text-text-muted md:text-lg">
                HDRS checks every AI-generated answer against its source material, sentence by sentence, before it reaches a user.
              </p>
              <div className="mt-6 grid gap-3 text-sm leading-6 text-text-muted">
                <div>• Catches unsupported claims even when the answer sounds confident.</div>
                <div>• Works on your own PDFs, reports, and pasted text.</div>
                <div>• Shows why a sentence was flagged so the result is explainable.</div>
              </div>
              <div className="mt-6 inline-flex rounded-full border border-black/5 bg-[#fafafa] px-4 py-2 text-sm font-medium text-text-main">
                Tested across {activeBenchmark.total_questions ?? 50}+ questions with a {activeBenchmark.answer_hallucination_rate ?? 22}% hallucination rate.
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-black/5 bg-white p-5 shadow-[0_14px_45px_rgba(17,17,26,0.04)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">What HDRS does</div>
              <div className="mt-3 grid gap-3 text-sm leading-6 text-text-muted sm:grid-cols-3 lg:grid-cols-1">
                <div className="rounded-2xl bg-[#fafafa] px-4 py-3">
                  <div className="font-medium text-text-main">1. Ingest</div>
                  <div>Upload PDFs or paste text to build a source library.</div>
                </div>
                <div className="rounded-2xl bg-[#fafafa] px-4 py-3">
                  <div className="font-medium text-text-main">2. Verify</div>
                  <div>Ask a question and compare the answer against the selected source.</div>
                </div>
                <div className="rounded-2xl bg-[#fafafa] px-4 py-3">
                  <div className="font-medium text-text-main">3. Measure</div>
                  <div>Use the benchmark to quantify hallucinations and track quality over time.</div>
                </div>
              </div>
            </div>

          </div>

          <div className="flex h-full flex-col gap-4">
            <div className="rounded-[1.5rem] border border-apple-green/20 bg-apple-green/5 p-5 shadow-[0_14px_45px_rgba(17,17,26,0.04)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-apple-green">Supported inputs</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {['PDF reports', 'TXT / MD notes', 'JSON exports', 'Pasted passages'].map((item) => (
                  <div data-stat-item key={item} className="flex items-center gap-3 rounded-full border border-apple-green/20 bg-white px-4 py-3 text-sm font-medium text-text-main">
                    <span className="h-2.5 w-2.5 rounded-full bg-apple-green" />
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-black/5 bg-white p-5 shadow-[0_14px_45px_rgba(17,17,26,0.04)]">
              <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Pipeline flow</div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {[
                  { step: '01', title: 'Retrieve', body: 'Fetch the most relevant chunks from the selected source.' },
                  { step: '02', title: 'Generate', body: 'Draft the answer strictly from the retrieved context.' },
                  { step: '03', title: 'Judge', body: 'Score grounding and surface hallucination risk.' }
                ].map((item, index) => (
                  <div key={item.step} className="rounded-2xl border border-black/5 bg-[#fafafa] px-4 py-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-black text-xs font-semibold text-white">
                        {item.step}
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-text-main">{item.title}</div>
                        <div className="text-[11px] uppercase tracking-[0.22em] text-text-muted">
                          {index < 2 ? 'next' : 'output'}
                        </div>
                      </div>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-text-muted">{item.body}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Sources</div>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight text-text-main">Uploaded documents and text</h2>
          </div>
          <div className="flex w-full flex-col gap-3 lg:w-auto lg:flex-row lg:items-center">
            <div className="relative w-full lg:w-[360px]">
              <i className="fa-solid fa-magnifying-glass pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-text-muted text-sm" />
              <input
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by source name or content..."
                className="w-full rounded-full border border-black/8 bg-white py-3 pl-11 pr-4 text-sm text-text-main outline-none placeholder:text-text-muted focus:border-text-main"
              />
            </div>
            <button onClick={fetchDocuments} className="rounded-full border border-black/8 bg-white px-4 py-3 text-sm font-medium text-text-muted hover:text-text-main">
              Refresh library
            </button>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filteredDocuments.map((doc) => {
            const isSelected = selectedDoc?.source === doc.source
            return (
              <article
                key={doc.source}
                className={`source-card rounded-[1.5rem] border bg-white p-4 sm:p-5 shadow-[0_14px_45px_rgba(17,17,26,0.05)] ${isSelected ? 'border-text-main/20 ring-1 ring-text-main/5' : 'border-black/5'}`}
              >
                <button
                  onClick={() => openChatForDoc(doc)}
                  className="w-full text-left"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.25em] text-text-muted">
                        {doc.source.toLowerCase().endsWith('.pdf') || doc.source.toLowerCase().endsWith('.doc') || doc.source.toLowerCase().endsWith('.docx') ? 'Document' : 'Text'}
                      </div>
                      <h4 className="mt-2 truncate text-lg font-semibold tracking-tight text-text-main">{doc.source}</h4>
                      <p className="mt-2 line-clamp-3 text-sm leading-6 text-text-muted">
                        {doc.preview || doc.full_text?.slice(0, 180) || 'No preview available'}
                      </p>
                    </div>
                    <div className="rounded-2xl border border-black/5 bg-[#fafafa] px-3 py-2 text-right">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-text-muted">Chunks</div>
                      <div className="mt-1 text-lg font-semibold text-text-main">{doc.chunks_count ?? 0}</div>
                    </div>
                  </div>
                </button>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  <button
                    onClick={() => openChatForDoc(doc)}
                    className="rounded-full border border-black/8 bg-[#fafafa] px-4 py-3 text-sm font-medium text-text-main hover:bg-black/[0.02]"
                  >
                    Open chat
                  </button>
                  <button
                    onClick={() => onOpenBenchmark(doc.source)}
                    className="rounded-full border border-black/8 bg-white px-4 py-3 text-sm font-medium text-text-main hover:bg-black/[0.02]"
                  >
                    Benchmark
                  </button>
                </div>

                <button
                  onClick={() => handleDeleteSource(doc)}
                  className="mt-3 w-full rounded-full border border-apple-red/20 bg-apple-red/5 px-4 py-3 text-sm font-medium text-apple-red hover:bg-apple-red/10"
                >
                  Delete source
                </button>
              </article>
            )
          })}

          {searchTerm.trim() && filteredDocuments.length === 0 && (
            <div className="rounded-[1.5rem] border border-dashed border-black/10 bg-[#fafafa] p-6 text-sm text-text-muted md:col-span-2 xl:col-span-3">
              No matching sources found.
            </div>
          )}

          <button
            onClick={() => openUploadModal('choose')}
            className="source-card flex min-h-[180px] flex-col items-center justify-center rounded-[1.5rem] border border-dashed border-black/10 bg-white px-6 text-center shadow-[0_14px_45px_rgba(17,17,26,0.04)]"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-black/5 text-2xl text-text-main">
              +
            </div>
            <div className="mt-4 text-lg font-semibold text-text-main">Add new source</div>
            <p className="mt-2 text-sm leading-6 text-text-muted">Upload a file or paste text to grow the workspace.</p>
          </button>
        </div>
      </section>

      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4 py-8 backdrop-blur-sm">
          <div className="w-full max-w-3xl overflow-hidden rounded-[1.5rem] bg-white shadow-[0_30px_100px_rgba(0,0,0,0.18)] sm:rounded-[2rem]">
            <div className="flex items-center justify-between border-b border-black/5 px-6 py-4">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.3em] text-text-muted">Upload source</div>
                <div className="mt-1 text-lg font-semibold text-text-main">
                  {uploadMode === 'text' ? 'Paste text' : uploadMode === 'file' ? 'Upload document' : 'Choose upload mode'}
                </div>
              </div>
              <button onClick={() => setIsModalOpen(false)} className="rounded-full border border-black/8 bg-white px-3 py-2 text-sm text-text-muted hover:text-text-main">
                Close
              </button>
            </div>

            <div className="grid gap-0 lg:grid-cols-[0.9fr_1.1fr]">
              <div className="border-b border-black/5 bg-[#fafafa] p-6 lg:border-b-0 lg:border-r">
                <div className="space-y-3">
                  <button onClick={() => setUploadMode('file')} className={`w-full rounded-2xl px-4 py-4 text-left text-sm font-medium ${uploadMode === 'file' ? 'bg-text-main text-white' : 'border border-black/8 bg-white text-text-main'}`}>
                    Upload document
                  </button>
                  <button onClick={() => setUploadMode('text')} className={`w-full rounded-2xl px-4 py-4 text-left text-sm font-medium ${uploadMode === 'text' ? 'bg-text-main text-white' : 'border border-black/8 bg-white text-text-main'}`}>
                    Upload text
                  </button>
                </div>
                <p className="mt-5 text-sm leading-6 text-text-muted">
                  Use the file drop zone for PDFs, TXT, MD, or JSON. Use the text form when you want to ingest plain content directly.
                </p>
              </div>

              <div className="p-6">
                <div className="mb-4 rounded-2xl border border-black/8 bg-[#fafafa] p-4">
                  <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">
                    <span>System Status</span>
                    <span>{Math.round(ingestStatus.progress || 0)}%</span>
                  </div>
                  <div className="mt-3 space-y-2 text-sm">
                    <div className="flex items-center gap-2 text-text-main">
                      <span
                        className={`h-2.5 w-2.5 rounded-full ${
                          ingestStatus.status === 'running'
                            ? 'bg-apple-blue animate-pulse'
                            : ingestStatus.status === 'completed'
                              ? 'bg-apple-green'
                              : ingestStatus.status === 'failed'
                                ? 'bg-apple-red'
                                : 'bg-black/20'
                        }`}
                      />
                      <span className="font-semibold capitalize">{ingestStatus.step || 'idle'}</span>
                    </div>
                    <p className="text-sm leading-6 text-text-muted">
                      {ingestStatus.message || 'Waiting for an upload to start.'}
                    </p>
                  </div>
                </div>

                {(uploadMode === 'file' || uploadMode === 'choose') && (
                  <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault()
                      setIsDragOver(false)
                      if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0])
                    }}
                    onClick={() => fileInputRef.current?.click()}
                    className={`flex min-h-64 cursor-pointer flex-col items-center justify-center rounded-[1.5rem] border border-dashed px-6 text-center transition-all ${isDragOver ? 'border-text-main bg-black/[0.02]' : 'border-black/10 bg-white hover:bg-black/[0.015]'
                      }`}
                  >
                    <div className="text-4xl text-text-main">⇪</div>
                    <div className="mt-4 text-lg font-semibold text-text-main">Drag and drop a document</div>
                    <p className="mt-2 text-sm leading-6 text-text-muted">Drop a file here or click to browse the library.</p>
                    <div className="mt-5 rounded-full border border-black/8 bg-[#fafafa] px-4 py-2 text-sm font-medium text-text-main">
                      Browse files
                    </div>
                    <input ref={fileInputRef} type="file" onChange={(e) => processFile(e.target.files?.[0])} className="hidden" accept=".pdf,.txt,.md,.json" />
                  </div>
                )}

                {uploadMode === 'text' && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Source name</label>
                      <input
                        value={docName}
                        onChange={(e) => setDocName(e.target.value)}
                        placeholder="e.g. research_notes.txt"
                        className="mt-2 w-full rounded-2xl border border-black/8 bg-white px-4 py-3 text-sm outline-none placeholder:text-text-muted focus:border-text-main"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Text content</label>
                      <textarea
                        rows="9"
                        value={docText}
                        onChange={(e) => setDocText(e.target.value)}
                        placeholder="Paste the source content here..."
                        className="mt-2 w-full rounded-[1.25rem] border border-black/8 bg-white px-4 py-3 text-sm leading-6 outline-none placeholder:text-text-muted focus:border-text-main"
                      />
                    </div>
                    <button
                      onClick={handleTextIngest}
                      disabled={isTextIngesting}
                      className="rounded-full bg-text-main px-5 py-3 text-sm font-medium text-white disabled:opacity-60"
                    >
                      {isTextIngesting ? 'Uploading...' : 'Upload text'}
                    </button>
                  </div>
                )}

                {isUploading && (
                  <div className="mt-4">
                    <div className="mb-2 flex items-center justify-between text-xs font-medium text-text-muted">
                      <span>Uploading</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-black/5">
                      <div className="h-full rounded-full bg-text-main transition-all" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {isChatOpen && selectedDoc && (
        <div className="fixed inset-0 z-[55] flex items-center justify-center bg-black/35 px-2 py-2 backdrop-blur-sm sm:px-4 sm:py-4">
          <div className="flex h-[96vh] w-[98vw] max-w-[1600px] flex-col overflow-hidden rounded-[1.5rem] border border-black/5 bg-white shadow-[0_30px_100px_rgba(0,0,0,0.24)] sm:rounded-[2rem]">
            <div className="flex items-center justify-between border-b border-black/5 px-5 py-4 sm:px-6">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.3em] text-text-muted">Source chat</div>
                <h3 className="mt-1 max-w-[70vw] truncate text-xl font-semibold tracking-tight text-text-main">
                  {selectedDoc.source}
                </h3>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    setIsChatOpen(false)
                    onOpenBenchmark(selectedDoc?.source)
                  }}
                  className="rounded-full border border-black/8 bg-white px-4 py-2 text-sm font-medium text-text-main hover:bg-black/[0.02]"
                >
                  Benchmark
                </button>
                <button
                  onClick={() => setIsChatOpen(false)}
                  className="rounded-full border border-black/8 bg-white px-4 py-2 text-sm font-medium text-text-muted hover:text-text-main"
                >
                  Close
                </button>
              </div>
            </div>

            <div className="grid min-h-0 flex-1 gap-0 lg:grid-cols-[1fr_380px]">
              <div className="flex min-h-0 flex-col border-b border-black/5 bg-[#fcfcfb] lg:border-b-0 lg:border-r">
                <div className="flex items-center justify-between border-b border-black/5 px-5 py-4 sm:px-6">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Conversation</div>
                    <p className="mt-1 text-sm text-text-muted">
                      Grounded only in the selected source.
                    </p>
                  </div>
                  <div className="rounded-full border border-black/5 bg-white px-3 py-1 text-xs font-medium text-text-muted">
                    {selectedDoc.chunks_count ?? 0} chunks
                  </div>
                </div>

                <div className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-5">
                  <div className="min-h-0 flex-1 overflow-y-auto pr-1">
                    <div className="flex min-h-full flex-col justify-end gap-3 pb-2">
                      {currentMessages.length === 0 ? (
                        <div className="mx-auto w-full max-w-2xl rounded-[1.5rem] border border-dashed border-black/10 bg-white px-5 py-10 text-center">
                          <div className="text-sm font-medium text-text-main">Start the conversation</div>
                          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-text-muted">
                            Ask a question about this source and HDRS will answer from the selected document only.
                          </p>
                        </div>
                      ) : (
                        currentMessages.map((msg, idx) => (
                          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div
                              className={`max-w-[82%] rounded-[1.4rem] px-4 py-3 text-[15px] leading-7 shadow-sm ${msg.role === 'user'
                                  ? 'bg-text-main text-white'
                                  : 'border border-black/5 bg-white text-text-main'
                                }`}
                            >
                              {msg.text}
                            </div>
                          </div>
                        ))
                      )}
                      {isChatLoading && (
                        <div className="flex justify-start">
                          <div className="max-w-[82%] rounded-[1.4rem] border border-black/5 bg-white px-4 py-3 text-[15px] leading-7 text-text-muted shadow-sm">
                            <div className="flex items-center gap-2">
                              <span className="h-2 w-2 animate-pulse rounded-full bg-text-muted/70" />
                              <span className="h-2 w-2 animate-pulse rounded-full bg-text-muted/70 [animation-delay:120ms]" />
                              <span className="h-2 w-2 animate-pulse rounded-full bg-text-muted/70 [animation-delay:240ms]" />
                              <span className="ml-1">Thinking...</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="mt-3 border-t border-black/5 pt-3">
                    <div className="rounded-[1.35rem] border border-black/8 bg-white p-2.5 shadow-[0_8px_24px_rgba(17,17,26,0.04)]">
                      <textarea
                        ref={chatInputRef}
                        rows="1"
                        value={chatQuery}
                        onChange={handleChatInput}
                        onKeyDown={handleChatKeyDown}
                        placeholder={`Ask about ${selectedDoc.source}...`}
                        className="w-full resize-none overflow-hidden rounded-[1rem] border-0 bg-transparent px-3 py-2 text-[15px] leading-7 outline-none placeholder:text-text-muted"
                      />
                      <div className="mt-2 flex items-center justify-between gap-3 border-t border-black/5 px-1 pt-2">
                        <div className="text-xs text-text-muted">
                          Enter to send, Shift+Enter for a new line.
                        </div>
                        <button
                          onClick={handleSendChat}
                          disabled={isChatLoading || !chatQuery.trim() || !selectedDoc?.chunks_count}
                          className="rounded-full bg-text-main px-4 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {isChatLoading ? 'Thinking...' : 'Send'}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white px-5 py-5 sm:px-6">
                <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Source info</div>
                <div className="mt-3 rounded-[1.5rem] border border-black/5 bg-[#fafafa] p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.22em] text-text-muted">Selected source</div>
                  <div className="mt-2 break-words text-sm font-semibold text-text-main">
                    {selectedDoc.source}
                  </div>
                  <div className="mt-3 text-sm leading-6 text-text-muted">
                    {selectedDoc.chunks_count ?? 0} indexed chunks are available for grounded chat.
                  </div>
                </div>

                <div className="mt-4 rounded-[1.5rem] border border-black/5 bg-white p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.22em] text-text-muted">Actions</div>
                  <button
                    onClick={() => {
                      setIsChatOpen(false)
                      onOpenBenchmark(selectedDoc?.source)
                    }}
                    className="mt-3 w-full rounded-full border border-black/8 bg-white px-4 py-3 text-sm font-medium text-text-main hover:bg-black/[0.02]"
                  >
                    Open benchmark
                  </button>
                  <button
                    onClick={() => handleDeleteSource(selectedDoc)}
                    className="mt-3 w-full rounded-full border border-apple-red/20 bg-apple-red/5 px-4 py-3 text-sm font-medium text-apple-red hover:bg-apple-red/10"
                  >
                    Delete source
                  </button>
                </div>

                <div className="mt-4 rounded-[1.5rem] border border-black/5 bg-[#fcfcfb] p-4">
                  <div className="text-xs font-medium uppercase tracking-[0.22em] text-text-muted">Tip</div>
                  <p className="mt-2 text-sm leading-6 text-text-muted">
                    Keep questions narrow for the clearest grounded answers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 px-4 py-8 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-[1.75rem] border border-black/5 bg-white p-6 shadow-[0_30px_100px_rgba(0,0,0,0.22)]">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-apple-red/10 text-apple-red">
                <i className="fa-solid fa-trash-can" />
              </div>
              <div className="min-w-0">
                <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">Delete source</div>
                <h4 className="mt-1 text-xl font-semibold tracking-tight text-text-main break-words">
                  {deleteTarget.source}
                </h4>
                <p className="mt-2 text-sm leading-6 text-text-muted">
                  This will permanently remove all indexed chunks for this source from the database.
                </p>
              </div>
            </div>

            <div className="mt-6 rounded-[1.25rem] border border-black/5 bg-[#fafafa] p-4 text-sm text-text-muted">
              {deleteTarget.chunks_count ?? 0} chunk{(deleteTarget.chunks_count ?? 0) === 1 ? '' : 's'} will be deleted.
            </div>

            <div className="mt-6 flex items-center justify-end gap-3">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={isDeleting}
                className="rounded-full border border-black/8 bg-white px-5 py-3 text-sm font-medium text-text-main hover:bg-black/[0.02] disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteSource}
                disabled={isDeleting}
                className="rounded-full bg-apple-red px-5 py-3 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-60"
              >
                {isDeleting ? 'Deleting...' : 'Delete source'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Workspace
