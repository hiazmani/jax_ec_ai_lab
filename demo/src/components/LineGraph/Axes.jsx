// src/components/LineGraph/Axes.jsx
import * as d3 from 'd3';

const Axes = ({ g, innerWidth, innerHeight, xScale, yScale, xLabel, yLabel, title, xTicks }) => {
    // X Axis
    g.append('g')
        .attr('transform', `translate(0, ${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(xTicks));

    // Y Axis
    g.append('g').call(d3.axisLeft(yScale));

    // X Axis Label
    g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 40)
        .attr('text-anchor', 'middle')
        .text(xLabel);

    // Y Axis Label
    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -50)
        .attr('text-anchor', 'middle')
        .text(yLabel);
};

export default Axes;
