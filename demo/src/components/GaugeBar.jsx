import React from 'react';

/**
 * GaugeBar: horizontal bar indicating a metric between 0 and 1.
 * Props:
 * - value: number between 0 and 1
 * - width: total width in px
 * - height: bar height in px
 * - thresholds?: Array of numbers for tick marks (e.g., [0,0.5,1])
 */
const GaugeBar = ({ value, width = 400, height = 20, thresholds = [0, 0.5, 1] }) => {
  const fillWidth = Math.max(0, Math.min(1, value)) * width;

  return (
    <svg width={width} height={height + 30}>
      {/* Background bar */}
      <rect x={0} y={10} width={width} height={height} fill="#e5e7eb" />
      {/* Filled portion */}
      <rect x={0} y={10} width={fillWidth} height={height} fill="#6366f1" />
      {/* Threshold ticks and labels */}
      {thresholds.map((t, i) => {
        const x = t * width;
        return (
          <g key={i}>
            <line x1={x} y1={10} x2={x} y2={10 + height} stroke="#4b5563" />
            <text x={x} y={10 + height + 15} textAnchor="middle" fontSize="10" fill="#374151">{t}</text>
          </g>
        );
      })}
      {/* Value label */}
      <text x={fillWidth} y={10 - 2} textAnchor="middle" fontSize="12" fill="#111827">{value.toFixed(2)}</text>
    </svg>
  );
};

export default GaugeBar;
