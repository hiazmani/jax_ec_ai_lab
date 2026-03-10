import React from 'react';

/**
 * MultiBarChart: grouped bar chart for comparing multiple series.
 * Props:
 * - groups: Array of group labels (x-axis categories)
 * - dataSets: Array of { label: string, values: number[], color: string }
 * - width: number (px)
 * - height: number (px)
 * - roundValues: boolean (default false)
 */
const MultiBarChart = ({ groups, dataSets, width = 600, height = 400, roundValues = false }) => {
  if (!groups || !dataSets || groups.length === 0 || dataSets.length === 0) return null;

  // Compute maximum value
  const maxValue = Math.max(...dataSets.flatMap(ds => ds.values));

  // Margins for axes
  const margin = { top: 20, right: 20, bottom: 50, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Generate y-axis ticks
  const tickCount = 5;
  const tickStep = maxValue / tickCount;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => i * tickStep);

  const groupCount = groups.length;
  const dataCount = dataSets.length;
  const groupWidth = chartWidth / groupCount;
  const barWidth = (groupWidth * 0.8) / dataCount;
  const barGap = dataCount > 1 ? (groupWidth * 0.8 - dataCount * barWidth) / (dataCount - 1) : 0;

  return (
    <svg width={width} height={height}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Y-axis */}
        <line x1={0} y1={0} x2={0} y2={chartHeight} stroke="#333" />
        {/* X-axis */}
        <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#333" />

        {/* Horizontal grid lines & y-axis ticks */}
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

        {/* Bars and labels */}
        {groups.map((group, i) => {
          const xGroup = i * groupWidth + (groupWidth * 0.2) / 2;
          return (
            <g key={group}>
              {dataSets.map((ds, j) => {
                const value = ds.values[i] || 0;
                const barHeight = (value / maxValue) * chartHeight;
                const x = xGroup + j * (barWidth + barGap);
                const y = chartHeight - barHeight;
                const color = ds.color;
                const displayVal = roundValues ? Math.round(value) : parseFloat(value.toFixed(1));
                return (
                  <g key={`${ds.label}-${i}`}>
                    <rect x={x} y={y} width={barWidth} height={barHeight} fill={color} />
                    <text
                      x={x + barWidth / 2}
                      y={y - 5}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#111"
                    >
                      {displayVal.toLocaleString()}
                    </text>
                  </g>
                );
              })}
              {/* Group category label */}
              <text
                x={xGroup + (dataCount * (barWidth + barGap) - barGap) / 2}
                y={chartHeight + 20}
                textAnchor="middle"
                fontSize="10"
                fill="#333"
              >
                {group}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
};

export default MultiBarChart;
