// src/components/LineGraph/Tooltip.jsx
import * as d3 from 'd3';

const Tooltip = ({ g, innerWidth, innerHeight, data, xScale, yScale, xScaleType, colorScale, unit }) => {
    // Create a group for the larger hover circles.
    const hoverGroup = g.append('g')
        .style('display', 'none');

    const hoverCircles = data.map((_, idx) =>
        hoverGroup.append('circle')
            .attr('r', 5)
            .attr('fill', colorScale(idx))
    );

    // Create a group for the tooltip.
    const tooltipGroup = g.append('g')
        .attr('class', 'tooltip-group')
        .style('display', 'none');

    // Inside tooltipGroup, create a subgroup for content.
    const tooltipContent = tooltipGroup.append('g')
        .attr('class', 'tooltip-content');

    // Insert a background rectangle (will be updated later)
    if (tooltipGroup.select('.tooltip-background').empty()) {
        tooltipGroup.insert('rect', ':first-child')
            .attr('class', 'tooltip-background');
    }

    // Add an overlay rectangle to capture mouse events.
    g.append('rect')
        .attr('class', 'overlay')
        .attr('width', innerWidth)
        .attr('height', innerHeight)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .on('mouseover', () => {
            hoverGroup.style('display', null);
            tooltipGroup.style('display', null);
        })
        .on('mousemove', function(event) {
            const [mx] = d3.pointer(event, this);
            const xValue = xScale.invert(mx);

            // For each line, find the closest data point.
            const lineTooltipData = data.map((lineData, idx) => {
                const points = lineData.points;
                const bisect = d3.bisector(d => xScaleType === 'time' ? new Date(d.x) : d.x).left;
                let i = bisect(points, xValue);
                i = Math.max(0, Math.min(i, points.length - 1));
                const d0 = points[i];
                // Update hover circle positions.
                const cx = xScale(xScaleType === 'time' ? new Date(d0.x) : d0.x);
                const cy = yScale(d0.y);
                hoverCircles[idx].attr('cx', cx).attr('cy', cy);
                return {
                    color: colorScale(idx),
                    xValue: d0.x,
                    yValue: d0.y,
                };
            });

            // Position tooltipGroup near the mouse.
            const [tooltipX, tooltipY] = d3.pointer(event, g.node());
            tooltipGroup.attr('transform', `translate(${tooltipX}, ${tooltipY - 40})`);

            const lineHeight = 15;
            const selection = tooltipContent.selectAll('.tooltip-line')
                .data(lineTooltipData);

            const enter = selection.enter().append('g')
                .attr('class', 'tooltip-line');

            // Append small circle.
            enter.append('circle')
                .attr('r', 3);

            // Append text.
            enter.append('text')
                .attr('x', 10)
                .attr('dy', '0.35em')
                .style('font-size', '12px');

            selection.merge(enter)
                .attr('transform', (d, i) => `translate(0, ${i * lineHeight})`)
                .each(function(d) {
                    const row = d3.select(this);
                    row.select('circle').attr('fill', d.color);
                    row.select('text')
                        .style('fill', d.color)
                        .text(`${d.xValue}: ${d.yValue.toFixed(2)} ${unit}`);
                });

            selection.exit().remove();

            // Compute bounding box from tooltipContent and update background.
            const bbox = tooltipContent.node().getBBox();
            tooltipGroup.select('.tooltip-background')
                .attr('x', bbox.x - 5)
                .attr('y', bbox.y - 5)
                .attr('width', bbox.width + 10)
                .attr('height', bbox.height + 10)
                .attr('fill', 'rgba(255,255,255,0.7)')
                .attr('stroke', 'black')
                .attr('stroke-width', 1);
        })
        .on('mouseleave', () => {
            hoverGroup.style('display', 'none');
            tooltipGroup.style('display', 'none');
        });
};

export default Tooltip;
