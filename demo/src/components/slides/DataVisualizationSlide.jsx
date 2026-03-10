import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph';
import * as d3 from 'd3';


const datasets = import.meta.glob('../../data/X_Datasets/*.json');

const DataVisualizationSlide = ({ goToPrev }) => {
    const [datasetNames, setDatasetNames] = useState([]);
    const [selectedDataset, setSelectedDataset] = useState('');
    const [data, setData] = useState([]);
    const [metadata, setMetadata] = useState({});
    const [viewMode, setViewMode] = useState('raw');

    useEffect(() => {
        const names = Object.keys(datasets).map(path =>
            path.replace('../../data/', '').replace('.json', '')
        );
        setDatasetNames(names);
        setSelectedDataset(names[0]);
    }, []);

    useEffect(() => {
        if (!selectedDataset) return;

        const loadData = async () => {
            const dataset = await datasets[`../../data/X_Datasets/${selectedDataset}.json`]();
            const { households, temporal, metadata } = dataset;
            const timestamps = temporal.timestamps.map(ts => new Date(ts));
            const dayOfWeek = temporal.day_of_week;

            const householdData = households.map(household => ({
                id: household.id,
                data: timestamps.map((timestamp, i) => ({
                    timestamp,
                    consumption: household.consumption[i],
                    production: household.production[i],
                    dayOfWeek: dayOfWeek[i],
                })),
            }));

            setMetadata(metadata);
            setData(householdData);
        };

        loadData();
    }, [selectedDataset]);

    // Data aggregation function
    const aggregateDailyData = (household) => {
        const grouped = d3.groups(household.data, d => d.dayOfWeek);
        return grouped.map(([day, entries]) => ({
            dayOfWeek: day,
            avgConsumption: d3.mean(entries, d => d.consumption),
            avgProduction: d3.mean(entries, d => d.production),
        })).sort((a, b) => a.dayOfWeek - b.dayOfWeek);
    };

    return (
        <div className="slide visualization-slide">
            <h1 className="slide-title">{'Data Visualization'}</h1>
            <p>{metadata.description}</p>

            <div className="selector-container">
                <label>Select Dataset:&nbsp;</label>
                <select
                    value={selectedDataset}
                    onChange={e => setSelectedDataset(e.target.value)}
                >
                    {datasetNames.map(name => (
                        <option key={name} value={name}>{name}</option>
                    ))}
                </select>

                <button onClick={() => setViewMode(viewMode === 'raw' ? 'dailyAvg' : 'raw')}>
                    View: {viewMode === 'raw' ? 'Raw Data' : 'Daily Averages'}
                </button>
            </div>

            {data.map(household => {
                const plotData = viewMode === 'raw'
                    ? [
                        {
                            label: 'Consumption',
                            points: household.data.map(d => ({ x: d.timestamp, y: d.consumption })),
                        },
                        {
                            label: 'Production',
                            points: household.data.map(d => ({ x: d.timestamp, y: d.production })),
                        },
                    ]
                    : [
                        {
                            label: 'Avg Consumption',
                            points: aggregateDailyData(household).map(d => ({
                                x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d.dayOfWeek],
                                y: d.avgConsumption,
                            })),
                        },
                        {
                            label: 'Avg Production',
                            points: aggregateDailyData(household).map(d => ({
                                x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d.dayOfWeek],
                                y: d.avgProduction,
                            })),
                        },
                    ];

                return (
                    <div key={household.id} className="graph-container">
                        <LineGraph
                            data={plotData}
                            width={700}
                            height={300}
                            xLabel={viewMode === 'raw' ? 'Time' : 'Day of Week'}
                            yLabel="Energy (kWh)"
                            title={`Household: ${household.id}`}
                            xScaleType={viewMode === 'raw' ? 'time' : 'linear'}
                        />
                    </div>
                );
            })}

            {/*<button onClick={goToPrev}>← Go Back</button>*/}
        </div>
    );
};

export default DataVisualizationSlide;
