import React, { useState, useRef } from 'react'
import axios from 'axios'

function ControlCenter({ status, fetchStatus, showToast, onLibraryEmpty, onBenchmarkReset, setLoadingOverlay }) {
  const [ingestTab, setIngestTab] = useState('upload')
  const [docName, setDocName] = useState('')
  const [docText, setDocText] = useState('')
  const [isTextIngesting, setIsTextIngesting] = useState(false)

  // File upload state
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadingFilename, setUploadingFilename] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef(null)

  // Handle text area ingestion
  const handleTextIngest = async () => {
    if (!docName.trim() || !docText.trim()) {
      showToast('Document Name and Content cannot be empty!', 'warning')
      return
    }

    setIsTextIngesting(true)
    setLoadingOverlay?.({ show: true, message: 'Uploading text into the database...', progress: 0 })
    try {
      const res = await axios.post('/api/ingest/text', {
        text: docText,
        source_name: docName
      })
      showToast(`Ingested successfully! Added ${res.data.chunks_added} chunks.`, 'success')
      setDocName('')
      setDocText('')
      fetchStatus()
    } catch (e) {
      showToast(`Ingestion failed: ${e.response?.data?.detail || e.message}`, 'error')
    } finally {
      setIsTextIngesting(false)
      setLoadingOverlay?.({ show: false, message: '', progress: null })
    }
  }

  // Handle file drop zone click
  const handleZoneClick = () => {
    fileInputRef.current.click()
  }

  // Handle file select/drag-drop
  const processFile = async (file) => {
    if (!file) return

    setUploadingFilename(file.name)
    setIsUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post('/api/ingest/file', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percent)
          setLoadingOverlay?.({ show: true, message: 'Uploading file into the database...', progress: percent })
        }
      })
      showToast(`File uploaded! Ingested ${res.data.chunks_added} chunks.`, 'success')
      fetchStatus()
    } catch (e) {
      showToast(`Upload failed: ${e.response?.data?.detail || e.message}`, 'error')
    } finally {
      setTimeout(() => {
        setIsUploading(false)
        setUploadingFilename('')
        setUploadProgress(0)
      }, 1500)
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      processFile(e.target.files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0])
    }
  }

  // Clear Database
  const handleClearDb = async () => {
    if (!confirm('Are you sure you want to delete the database? This action clears all vector collections.')) return

    try {
      const res = await axios.post('/api/db/clear')
      showToast(res.data.message, 'success')
      if (res.data?.benchmark_reset) {
        onBenchmarkReset?.()
      }
      onLibraryEmpty?.()
      fetchStatus()
    } catch (e) {
      showToast(`Clear failed: ${e.response?.data?.detail || e.message}`, 'error')
    }
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      
      {/* Left: Document Ingestion */}
      <div className="lg:col-span-8 apple-panel p-4 sm:p-6">
        <div className="border-b border-border-light pb-4 mb-6">
          <h3 className="text-base font-bold text-text-main flex items-center gap-2">
            <i className="fa-solid fa-sliders text-apple-blue"></i> Document Ingestion
          </h3>
          <p className="text-[11px] text-text-muted mt-0.5 font-medium">Add text documents or PDF files to compile the RAG knowledge corpus</p>
        </div>

        {/* Ingest Tabs */}
        <div className="flex border-b border-border-light mb-6 gap-6">
          <button 
            onClick={() => setIngestTab('upload')}
            className={`flex items-center gap-2 pb-3 font-bold text-xs border-b-2 ${
              ingestTab === 'upload' 
                ? 'border-apple-blue text-apple-blue' 
                : 'border-transparent text-text-muted hover:text-text-main'
            }`}
          >
            <i className="fa-solid fa-file-pdf"></i> Upload File
          </button>
          
          <button 
            onClick={() => setIngestTab('paste')}
            className={`flex items-center gap-2 pb-3 font-bold text-xs border-b-2 ${
              ingestTab === 'paste' 
                ? 'border-apple-blue text-apple-blue' 
                : 'border-transparent text-text-muted hover:text-text-main'
            }`}
          >
            <i className="fa-solid fa-keyboard"></i> Paste Text
          </button>
        </div>

        {/* File Ingest Tab */}
        {ingestTab === 'upload' && (
          <div>
            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={handleZoneClick}
              className={`border border-dashed rounded-xl p-10 text-center cursor-pointer bg-base-light flex flex-col items-center justify-center ${
                isDragOver ? 'border-apple-blue bg-apple-blue/5' : 'border-border-accent hover:border-apple-blue/30'
              }`}
            >
              <i className="fa-solid fa-cloud-arrow-up text-3xl text-text-muted mb-3"></i>
              <h4 className="text-xs font-bold text-text-main mb-1">Drag & drop files here</h4>
              <p className="text-[10px] text-text-muted mb-4">Supports PDF, TXT, MD, and JSON files</p>
              <span className="px-4 py-2 rounded-lg border border-border-light bg-surface-light hover:bg-base-light text-xs font-semibold text-text-main select-none pointer-events-none shadow-sm">
                Browse Files
              </span>
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".pdf,.txt,.md,.json"
                className="hidden" 
              />
            </div>

            {isUploading && (
              <div className="mt-5 bg-surface-light border border-border-light p-4 rounded-xl shadow-sm">
                <div className="flex justify-between items-center text-xs font-bold mb-2">
                  <span className="text-text-main truncate max-w-[80%]">{uploadingFilename}</span>
                  <span className="text-apple-blue">{uploadProgress}%</span>
                </div>
                <div className="w-full h-1 bg-border-light rounded-full overflow-hidden">
                  <div 
                    style={{ width: `${uploadProgress}%` }}
                    className="h-full bg-apple-blue"
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Text Area Ingest Tab */}
        {ingestTab === 'paste' && (
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[9px] uppercase tracking-wider font-bold text-text-muted">Source/Document Name</label>
              <input 
                type="text" 
                value={docName}
                onChange={(e) => setDocName(e.target.value)}
                placeholder="e.g., Space_Exploration_History.txt" 
                className="w-full bg-surface-light border border-border-light rounded-lg text-xs text-text-main p-3 outline-none focus:border-apple-blue"
              />
            </div>
            
            <div className="flex flex-col gap-1.5">
              <label className="text-[9px] uppercase tracking-wider font-bold text-text-muted">Document Content</label>
              <textarea 
                rows="6" 
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
                placeholder="Paste the content of the article or document here..." 
                className="w-full bg-surface-light border border-border-light rounded-lg text-xs text-text-main p-3 outline-none focus:border-apple-blue resize-y leading-relaxed"
              ></textarea>
            </div>

            <button 
              onClick={handleTextIngest}
              disabled={isTextIngesting}
              className="bg-text-main text-white px-4 py-2.5 rounded-lg text-xs font-bold self-start hover:bg-text-main/90 flex items-center gap-2"
            >
              <i className="fa-solid fa-plus"></i> Ingest Document
            </button>
          </div>
        )}
      </div>

      {/* Right: Info Panel & Database Operations */}
      <div className="flex flex-col gap-6 lg:col-span-4">
        
        {/* System configuration status card */}
        <div className="apple-panel p-6 flex flex-col justify-between">
          <div className="border-b border-border-light pb-4 mb-4">
            <h3 className="text-base font-bold text-text-main flex items-center gap-2">
              <i className="fa-solid fa-shield-halved text-apple-blue"></i> System Config
            </h3>
          </div>

          <div className="flex flex-col gap-4 flex-grow justify-center mb-6">
            <div className="flex flex-col">
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider">Active Embedder</span>
              <span className="text-xs font-semibold mt-0.5 text-text-main">
                Gemini gemini-embedding-001 (3072 dims)
              </span>
            </div>
            
            <div className="flex flex-col">
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider">Active Generator</span>
              <span className="text-xs font-semibold mt-0.5 text-text-main">
                {status.active_generator || 'gemini-3.6-flash (API)'}
              </span>
              {status.ollama_available && (
                <span className="text-[10px] mt-0.5 text-apple-green font-semibold">
                  <i className="fa-solid fa-circle-check mr-1"></i>Local · No quota limits
                </span>
              )}
            </div>

            <div className="flex flex-col">
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider">Vector DB Engine</span>
              <span className="text-xs font-semibold text-text-main mt-0.5">{status.vector_store_type}</span>
            </div>

            <div className="flex flex-col">
              <span className="text-[9px] text-text-muted uppercase font-bold tracking-wider">API Key Configuration</span>
              <span className={`text-xs font-bold mt-0.5 ${
                status.gemini_api_key_configured ? 'text-apple-green' : 'text-apple-red'
              }`}>
                {status.gemini_api_key_configured ? 'Configured (Active)' : 'Missing GOOGLE_API_KEY'}
              </span>
            </div>
          </div>

          <div className="border-t border-border-light pt-5">
            <h4 className="text-xs font-bold text-apple-red mb-1">Database Maintenance</h4>
            <p className="text-[10px] text-text-muted mb-4 leading-normal font-medium">
              Deleting the database removes all ingested documents and vector indices.
            </p>
            <button 
              onClick={handleClearDb}
              className="w-full bg-apple-red hover:bg-red-600 text-white font-bold text-xs py-3 px-4 rounded-lg flex items-center justify-center gap-2 shadow-sm"
            >
              <i className="fa-solid fa-trash-can"></i> Clear Database
            </button>
          </div>
        </div>

      </div>

    </div>
  )
}

export default ControlCenter
