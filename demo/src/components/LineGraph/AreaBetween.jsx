// src/components/LineGraph/AreaBetween.jsx
import * as d3 from 'd3';

const AreaBetween = ({ g, data, xScale, yScale, xScaleType, areaColor, areaOpacity }) => {
    const [lineA, lineB] = data;
    const pointsA = lineA.points;
    const pointsB = lineB.points;

    if (pointsA.length === pointsB.length) {
        const areaData = pointsA.map((dA, i) => {
            const dB = pointsB[i];
            return {
                x: dA.x,
                y0: Math.min(dA.y, dB.y),
                y1: Math.max(dA.y, dB.y),
            };
        });

        const areaGenerator = d3.area()
            .x(d => xScale(xScaleType === 'time' ? new Date(d.x) : d.x))
            .y0(d => yScale(d.y0))
            .y1(d => yScale(d.y1))
            .curve(d3.curveMonotoneX);

        const path = g.append('path')
            .datum(areaData)
            .attr('fill', areaColor)
            .attr('fill-opacity', areaOpacity)
            .attr('d', areaGenerator);

        // Animate the area drawing (like a line)
        const totalLength = path.node().getTotalLength();

        path
            .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
            .attr('stroke-dashoffset', totalLength)
            .attr('stroke', areaColor)
            .attr('stroke-opacity', areaOpacity)
            .attr('fill-opacity', 0.0) // start invisible
            .transition()
            .duration(1000)
            .ease(d3.easeCubicInOut)
            .attr('stroke-dashoffset', 0)
            .attr('fill-opacity', areaOpacity) // fade fill in at end
            .on('end', () => {
                // cleanup stroke after animation
                path.attr('stroke', null);
            });
    }
};

export default AreaBetween;
