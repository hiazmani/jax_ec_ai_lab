import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import LineGraph from './LineGraph';
import energyProfile from '../data/energy_profile_ex.json';

const BatteryEffectGraph = () => {
    const [batterySize, setBatterySize] = useState(4); // in kWh
    const [graphData, setGraphData] = useState([]);
    const [extraAreas, setExtraAreas] = useState([]);
    const [gridUse, setGridUse] = useState(0);

    const consumption = energyProfile.data.find(p => p.legend_title.includes('Consumption')).y;
    const production = energyProfile.data.find(p => p.legend_title.includes('Production')).y;
    const x = energyProfile.data[0].x;

    useEffect(() => {
        let soc = 0;
        let max = batterySize;
        const charging = [];
        const discharging = [];
        const unmet = [];
        let totalGridDraw = 0;

        for (let i = 0; i < x.length; i++) {
            const need = consumption[i];
            const solar = production[i];

            const usedSolar = Math.min(solar, need);
            let remaining = need - usedSolar;

            const fromBattery = Math.min(remaining, soc);
            soc -= fromBattery;
            remaining -= fromBattery;

            const fromGrid = remaining;
            totalGridDraw += fromGrid;

            const excessSolar = solar - usedSolar;
            const toStore = Math.min(excessSolar, max - soc);
            soc += toStore;

            charging.push(toStore);
            discharging.push(fromBattery);
            unmet.push(fromGrid);
        }

        setGridUse(totalGridDraw);

        setExtraAreas([
            {
                label: 'Battery Charging',
                color: 'rgba(46, 204, 113, 0.3)',
                points: x.map((xi, i) => ({ x: xi, y: charging[i] }))
            },
            {
                label: 'Battery Discharging',
                color: 'rgba(52, 152, 219, 0.3)',
                points: x.map((xi, i) => ({ x: xi, y: discharging[i] }))
            },
            {
                label: 'Grid Draw',
                color: 'rgba(231, 76, 60, 0.3)',
                points: x.map((xi, i) => ({ x: xi, y: unmet[i] }))
            }
        ]);

        const lines = [
            {
                label: 'Energy Consumption',
                color: getComputedStyle(document.documentElement).getPropertyValue('--color-primaryBlue').trim(),
                points: x.map((xi, i) => ({ x: xi, y: consumption[i] })),
            },
            {
                label: 'Solar Production',
                color: getComputedStyle(document.documentElement).getPropertyValue('--color-primaryGreen').trim(),
                points: x.map((xi, i) => ({ x: xi, y: production[i] })),
            }
        ];

        setGraphData(lines);
    }, [batterySize]);

    return (
        <div className="w-full space-y-4">
            <div className="flex items-center gap-4">
                <label className="text-center" htmlFor="batterySize">Battery Capacity (kWh):</label>
                <input
                    id="batterySize"
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={batterySize}
                    onChange={(e) => setBatterySize(parseFloat(e.target.value))}
                />
                <span>{batterySize.toFixed(1)} kWh</span>
            </div>

            <div className="text-sm text-gray-700">
                Grid energy used: <strong>{gridUse.toFixed(2)} kWh</strong>
            </div>

            <LineGraph
                data={graphData}
                width={700}
                height={300}
                xLabel={energyProfile.x_label}
                yLabel={energyProfile.y_label}
                title={`Battery Effect at ${batterySize} kWh`}
                colors={graphData.map(d => d.color)}
                xScaleType="linear"
                xTicks={10}
                extraAreas={extraAreas}
            />
        </div>
    );
};

export default BatteryEffectGraph;
