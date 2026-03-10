// src/components/LineGraph/Lines.jsx
import * as d3 from 'd3';

const Lines = ({ g, data, lineGenerator, colorScale, width, margin }) => {
    data.forEach((lineData, idx) => {
        const path = g.append('path')
            .datum(lineData.points)
            .attr('fill', 'none')
            .attr('stroke', colorScale(idx))
            .attr('stroke-width', 2)
            .attr('d', lineGenerator);

        // Animate drawing the line
        const totalLength = path.node().getTotalLength();

        path
            .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
            .attr('stroke-dashoffset', totalLength)
            .transition()
            .duration(1000)
            .ease(d3.easeCubicInOut)
            .attr('stroke-dashoffset', 0);

        // Append legend elements outside g
        const parent = d3.select(g.node().parentNode);

        // Animate circle
                parent.append('circle')
                    .attr('cx', width) // start off-screen to the right
                    .attr('cy', margin.top + idx * 20)
                    .attr('r', 5)
                    .style('fill', colorScale(idx))
                    .transition()
                    .duration(400)
                    .delay(500 + idx * 150)
                    .ease(d3.easeCubicOut)
                    .attr('cx', width - margin.right - 150); // final position

        // Animate text
                parent.append('text')
                    .attr('x', width + 10) // also start off-screen
                    .attr('y', margin.top + idx * 20 + 5)
                    .text(lineData.label)
                    .style('font-size', '12px')
                    .style('opacity', 0)
                    .transition()
                    .duration(400)
                    .delay(500 + idx * 150 + 100) // stagger more
                    .ease(d3.easeCubicOut)
                    .style('opacity', 1)
                    .attr('x', width - margin.right - 140); // slide into place
    });
};

export default Lines;
