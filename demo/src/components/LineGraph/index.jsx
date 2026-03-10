// src/components/LineGraph/index.jsx
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import Axes from './Axes';
import Lines from './Lines';
import AreaBetween from './AreaBetween';
import Tooltip from './Tooltip';

const LineGraph = ({
                       data,
                       width,
                       height,
                       xLabel,
                       yLabel,
                       title,
                       colors,
                       xScaleType,
                       xTicks,
                       showAreaBetween,
                       areaColor = "red",
                       areaOpacity = 0.2,
                       activeIndices,  // array of indices (e.g. [0, 2]) that are active
                       extraAreas = []
                   }) => {
    const svgRef = useRef();

    useEffect(() => {
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // Always extract unit from yLabel, even if no line is active
        const unitMatch = yLabel.match(/\(([^)]+)\)$/);
        const unit = unitMatch ? unitMatch[1] : '';

        const margin = { top: 20, right: 10, bottom: 50, left: 70 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        // Compute scales based on the full dataset
        const allPoints = data.flatMap(line => line.points);
        const xScale = xScaleType === 'time'
            ? d3.scaleTime()
                .domain(d3.extent(allPoints, d => new Date(d.x)))
                .range([0, innerWidth])
            : d3.scaleLinear()
                .domain(d3.extent(allPoints, d => d.x))
                .range([0, innerWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(allPoints, d => d.y) * 1.2])
            .range([innerHeight, 0]);

        // Line generator (for drawing each line)
        const lineGenerator = d3.line()
            .x(d => xScale(xScaleType === 'time' ? new Date(d.x) : d.x))
            .y(d => yScale(d.y))
            .curve(d3.curveMonotoneX);

        // Color scale
        const colorScale = colors
            ? d3.scaleOrdinal().range(colors)
            : d3.scaleOrdinal(d3.schemeCategory10);

        // Main group
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Draw axes & labels via the Axes component.
        Axes({ g, innerWidth, innerHeight, xScale, yScale, xLabel, yLabel, title, xTicks });

        // Determine which lines are active.
        // If activeIndices is undefined or empty, assume all lines are active.
        const activeData = (!activeIndices || activeIndices.length === 0)
            ? data
            : data.filter((_, idx) => activeIndices.includes(idx));

        // Extract colors for active data
        const activeColors = activeData.map(d => d.color);

        // Add shaded areas if provided
        extraAreas.forEach((areaData) => {
            const areaGenerator = d3.area()
                .x(d => xScale(xScaleType === 'time' ? new Date(d.x) : d.x))
                .y0(() => yScale(0))
                .y1(d => yScale(d.y))
                .curve(d3.curveMonotoneX);

            g.append('path')
                .datum(areaData.points)
                .attr('fill', areaData.color)
                .attr('d', areaGenerator);
        });
        // LEGEND for extraAreas (if provided)
        extraAreas?.forEach((area, idx) => {
            svg.append('rect')
                .attr('x', width - margin.right - 150)
                .attr('y', margin.top + (data.length + idx) * 20 - 6)
                .attr('width', 10)
                .attr('height', 10)
                .attr('fill', area.color);

            svg.append('text')
                .attr('x', width - margin.right - 140)
                .attr('y', margin.top + (data.length + idx) * 20 + 3)
                .text(area.label)
                .style('font-size', '12px');
        });

        // Draw lines & legend for active lines only.
        Lines({ g, data: activeData, lineGenerator, colorScale: d3.scaleOrdinal().range(activeColors), width, margin });

        // Draw area between lines if enabled.
        if (showAreaBetween && data.length === 2) {
            AreaBetween({ g, data, xScale, yScale, xScaleType, areaColor, areaOpacity });
        }

        // Add tooltip (which may show data for all lines or only active lines)
        Tooltip({ g, innerWidth, innerHeight, data: activeData, xScale, yScale, xScaleType, colorScale, unit });

    }, [data, width, height, xLabel, yLabel, title, colors, xScaleType, xTicks, showAreaBetween, areaColor, areaOpacity, activeIndices]);

    return <svg ref={svgRef} width={width} height={height}></svg>;
};

export default LineGraph;
