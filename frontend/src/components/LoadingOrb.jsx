import React from 'react'

function LoadingOrb({ compact = false }) {
  return (
    <div
      className={`hdrs-loader ${compact ? 'hdrs-loader--compact' : ''}`}
      aria-label="Loading"
      role="status"
      style={compact ? { fontSize: '0.28rem' } : undefined}
    >
      <div className="hdrs-loader__up">
        <div className="hdrs-loader__loaders">
          {Array.from({ length: 10 }).map((_, idx) => (
            <div key={`bar-${idx}`} className="hdrs-loader__bar" />
          ))}
        </div>
        <div className="hdrs-loader__loaders-b">
          {Array.from({ length: 9 }).map((_, idx) => (
            <div key={`ball-${idx}`} className="hdrs-loader__rail">
              <div className={`hdrs-loader__ball hdrs-loader__ball--${idx}`} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default LoadingOrb
