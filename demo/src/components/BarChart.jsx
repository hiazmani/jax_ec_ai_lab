import React from 'react';

/**
 * BarChart: simple single-series bar chart with axes and optional rounding.
 * Props:
 * - data: Array of { label: string, value: number, color?: string }
 * - width: number (px)
 * - height: number (px)
 * - roundValues: boolean (default false) - whether to round displayed values and ticks.
 */
const BarChart = ({ data, width = 400, height = 300, roundValues = false }) => {
  if (!data || data.length === 0) return null;

  // Margins for axes
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const maxValue = Math.max(...data.map(d => d.value));

  // Generate y-axis ticks
  const tickCount = 5;
  const tickStep = maxValue / tickCount;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => i * tickStep);

  const barCount = data.length;
  const gap = chartWidth / barCount;
  const barWidth = gap * 0.6;

  return (
    <svg width={width} height={height}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Y-axis line */}
        <line x1={0} y1={0} x2={0} y2={chartHeight} stroke="#333" />
        {/* X-axis line */}
        <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#333" />

        {/* Horizontal grid lines and labels */}
        {ticks.map((t, i) => {
          const y = chartHeight - (t / maxValue) * chartHeight;
          const label = roundValues ? Math.round(t) : t.toFixed(0);
          return (
            <g key={i}>
              <line x1={0} y1={y} x2={chartWidth} y2={y} stroke="#ccc" strokeDasharray="2,2" />
              <text x={-8} y={y + 4} textAnchor="end" fontSize="10" fill="#333">
                {label.toLocaleString()}
              </text>
            </g>
          );
        })}

        {/* Bars and x-axis labels */}
        {data.map((d, i) => {
          const barHeight = (d.value / maxValue) * chartHeight;
          const x = i * gap + (gap - barWidth) / 2;
          const y = chartHeight - barHeight;
          const color = d.color || '#3b82f6';
          const displayVal = roundValues ? Math.round(d.value) : d.value;
          return (
            <g key={d.label}>
              <rect x={x} y={y} width={barWidth} height={barHeight} fill={color} />
              <text
                x={x + barWidth / 2}
                y={y - 5}
                textAnchor="middle"
                fontSize="10"
                fill="#111827"
              >
                {displayVal.toLocaleString()}
              </text>
              <text
                x={x + barWidth / 2}
                y={chartHeight + 14}
                textAnchor="middle"
                fontSize="10"
                fill="#333"
              >
                {d.label}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
};

export default BarChart;
