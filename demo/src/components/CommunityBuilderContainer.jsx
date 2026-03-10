import React, { useState, useEffect, useRef } from 'react';
import { Application } from '@pixi/react';
import { EnergyNetwork } from './EnergyNetwork.jsx';
import LineGraph from './LineGraph/index.jsx';

export default function CommunityBuilderContainer() {
    const [width, setWidth] = useState(window.innerWidth);
    const [height, setHeight] = useState(window.innerHeight);
    const [graphData, setGraphData] = useState([]);
    const energyNetworkRef = useRef(null);

    useEffect(() => {
        const handleResize = () => {
            setWidth(window.innerWidth);
            setHeight(window.innerHeight);
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    const clearConnections = () => {
        if (energyNetworkRef.current) {
            energyNetworkRef.current.clearAllConnections();
        }
    };

    return (
        <div className="w-full h-screen grid grid-cols-1 grid-cols-2 gap-8 items-start">
            <div>
                <Application width={width/2} height={height/2} backgroundAlpha={0}>
                    <EnergyNetwork
                        setGraphData={setGraphData}
                        ref={energyNetworkRef}
                        width={width/2}
                        height={height/2}
                    />
                </Application>
                <button
                    className="background-primaryBlue"
                    onClick={clearConnections}>Clear Connections</button>
            </div>
            <div>
                <p>
                    <strong>Community Energy Balance</strong>
                    <br />
                    This graph shows the energy consumption and production of the community over time.
                </p>
                <LineGraph
                    data={graphData}
                    width={700}
                    height={300}
                    xLabel="Time (hours)"
                    yLabel="Energy (kWh)"
                    title="Community Energy Balance"
                    xScaleType="linear"
                    xTicks={6}
                    colors={graphData.map(d => d.color)}
                />
            </div>
        </div>
    );
}
