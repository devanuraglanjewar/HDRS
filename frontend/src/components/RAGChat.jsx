import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function RAGChat({ showToast, setLoadingOverlay }) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [retrievedChunks, setRetrievedChunks] = useState([])
  const [topK, setTopK] = useState(3)
  const [modelName, setModelName] = useState('gemini-3.6-flash')
  const [isLoading, setIsLoading] = useState(false)
  const [expandedChunk, setExpandedChunk] = useState(null)

  const messagesEndRef = useRef(null)

  // Auto scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = async () => {
    const queryText = query.trim()
    if (!queryText) return

    // Append User message
    const userMsg = { role: 'user', text: queryText, timestamp: new Date().toLocaleTimeString() }
    setMessages(prev => [...prev, userMsg])
    setQuery('')
    setIsLoading(true)
    setLoadingOverlay?.({ show: true, message: 'Generating answer from retrieved context...', progress: null })

    try {
      const res = await axios.post('/api/ask', {
        query: queryText,
        top_k: parseInt(topK),
        model_name: modelName
      })

      // Append assistant message
      const assistantMsg = { 
        role: 'assistant', 
        text: res.data.answer, 
        timestamp: new Date().toLocaleTimeString(),
        isNew: true // trigger typewriter only once
      }
      
      // Update states
      setMessages(prev => [...prev, assistantMsg])
      setRetrievedChunks(res.data.retrieved_chunks || [])
    } catch (e) {
      const errorMsg = { 
        role: 'assistant', 
        text: `Error retrieving answer: ${e.response?.data?.detail || e.message}`, 
        timestamp: new Date().toLocaleTimeString() 
      }
      setMessages(prev => [...prev, errorMsg])
      setRetrievedChunks([])
      showToast('Error fetching RAG generation', 'error')
    } finally {
      setIsLoading(false)
      setLoadingOverlay?.({ show: false, message: '', progress: null })
    }
  }

  return (
    <div className="flex min-h-[calc(100vh-170px)] flex-col overflow-hidden rounded-2xl border border-border-dark bg-surface-dark/20 lg:h-[calc(100vh-170px)] lg:flex-row">
      
      {/* Left Chat Window */}
      <div className="flex flex-grow flex-col border-b border-border-dark bg-black/10 lg:border-b-0 lg:border-r">
        
        {/* Chat Configuration Bar */}
        <div className="flex flex-col gap-4 border-b border-border-dark bg-surface-dark/40 px-4 py-4 sm:flex-row sm:gap-6 sm:px-6">
          <div className="flex flex-1 flex-col gap-1.5 sm:max-w-[240px]">
            <label className="text-[10px] font-bold text-text-muted uppercase">Generator Model</label>
            <select 
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="bg-black/30 border border-border-dark rounded-lg text-xs text-text-main p-2 outline-none focus:border-primary-neon cursor-pointer"
            >
              <option value="gemini-3.6-flash">Gemini 3.6 Flash (Default)</option>
              <option value="gemini-2.5-pro">Gemini 2.5 Pro (Reasoning)</option>
            </select>
          </div>

          <div className="flex flex-1 flex-col gap-1.5 sm:max-w-[240px]">
            <label className="text-[10px] font-bold text-text-muted uppercase flex justify-between">
              <span>Top-K Chunks</span>
              <span className="text-primary-neon font-bold">{topK}</span>
            </label>
            <input 
              type="range" 
              min="1" 
              max="10" 
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className="w-full h-1.5 bg-border-dark rounded-lg appearance-none cursor-pointer accent-primary-neon mt-2"
            />
          </div>
        </div>

        {/* Chat Message feed */}
        <div className="flex-grow overflow-y-auto p-6 flex flex-col gap-5">
          {messages.length === 0 ? (
            <div className="self-center my-auto max-w-[90%] bg-surface-dark/45 border border-dashed border-border-dark p-6 rounded-xl text-center">
              <div className="flex gap-3 items-start text-text-muted">
                <i className="fa-solid fa-robot text-lg text-primary-neon mt-1"></i>
                <p className="text-xs leading-relaxed text-left">
                  Welcome to RAG baseline chat! Ask questions based on your ingested documents. The system is instructed to answer strictly using the retrieved chunks, showing baseline performance.
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`flex gap-4 max-w-[80%] ${msg.role === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}
              >
                <div className={`w-9 h-9 rounded-full flex items-center justify-center text-sm shrink-0 border shadow-lg ${
                  msg.role === 'user' 
                    ? 'bg-gradient-to-br from-primary-neon to-indigo-600 text-white border-primary-neon/20' 
                    : 'bg-surface-dark text-primary-neon border-border-dark'
                }`}>
                  <i className={`fa-solid ${msg.role === 'user' ? 'fa-user' : 'fa-robot'}`}></i>
                </div>
                
                <div className={`p-4 rounded-2xl shadow-md ${
                  msg.role === 'user'
                    ? 'bg-primary-neon/10 border border-primary-neon/30 text-white rounded-tr-none'
                    : 'bg-surface-dark border border-border-dark text-text-main rounded-tl-none'
                }`}>
                  <p className="text-[14.5px] leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                  <span className="text-[9px] text-text-muted mt-2 block text-right font-medium">{msg.timestamp}</span>
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div />
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Query Input Box */}
        <div className="p-5 bg-surface-dark/30 border-t border-border-dark flex gap-3">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
            placeholder="Ask a question about the corpus..." 
            disabled={isLoading}
            className="flex-grow bg-black/40 border border-border-dark rounded-full text-sm text-text-main px-6 py-3.5 outline-none focus:border-primary-neon"
          />
          <button 
            onClick={handleSendMessage}
            disabled={isLoading || !query.trim()}
            className="w-12 h-12 rounded-full bg-gradient-to-r from-primary-neon to-indigo-600 flex items-center justify-center text-white flex-shrink-0 disabled:opacity-50 disabled:pointer-events-none"
          >
            <i className="fa-solid fa-paper-plane"></i>
          </button>
        </div>
      </div>

      {/* Right Source Chunk Sidebar */}
      <aside className="flex w-full shrink-0 flex-col border-t border-border-dark bg-surface-dark/10 lg:w-80 lg:border-t-0">
        <div className="p-6 border-b border-border-dark bg-surface-dark/30">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <i className="fa-solid fa-brain text-primary-neon"></i> Retrieved Sources
          </h3>
          <p className="text-[10px] text-text-muted mt-1">Context chunks matched by embedding search</p>
        </div>

        <div className="flex-grow overflow-y-auto p-4 flex flex-col gap-4">
          {retrievedChunks.length === 0 ? (
            <div className="my-auto text-center p-6 text-text-dark flex flex-col items-center">
              <i className="fa-solid fa-magnifying-glass text-3xl mb-3 opacity-40"></i>
              <p className="text-xs leading-normal">
                Submit a query to view retrieved context chunks and similarity scores
              </p>
            </div>
          ) : (
            retrievedChunks.map((chunk, idx) => {
              const isExpanded = expandedChunk === idx
              return (
                <div 
                  key={idx}
                  onClick={() => setExpandedChunk(isExpanded ? null : idx)}
                  className={`retrieval-chunk-card bg-surface-dark/60 border rounded-lg p-4 cursor-pointer ${
                    isExpanded ? 'border-primary-neon/50 bg-surface-dark/90 shadow-md' : 'border-border-dark'
                  }`}
                >
                  <div className="flex justify-between items-center text-[10px] font-bold text-text-muted mb-2 border-b border-border-dark/35 pb-2">
                    <span className="text-primary-neon truncate max-w-[140px]" title={chunk.source}>
                      <i className="fa-solid fa-file-invoice mr-1"></i> {chunk.source}
                    </span>
                    <span className="text-warning-neon">Sim: {chunk.score}</span>
                  </div>
                  <p className={`text-xs text-text-main leading-relaxed ${
                    isExpanded ? '' : 'line-clamp-4'
                  }`}>
                    {chunk.text}
                  </p>
                </div>
              )
            })
          )}
        </div>
      </aside>

    </div>
  )
}

export default RAGChat
