import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProductionProfile from '../../data/energy_production_ex.json';
import PixiSplash from "../PixiSplash.jsx";

const SolarPower = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryGreen = getCSSVariable('--color-primaryGreen');
    const colors = [primaryGreen];

    useEffect(() => {
        const lineData = energyProductionProfile.data.map(line => ({
            label: line.legend_title,
            color: primaryGreen,
            points: line.x.map((x, i) => ({ x: x, y: line.y[i] })),
        }));
        setData(lineData);
    }, []);

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} />

            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">Solar Power: Energy from the Sun</h1>
                <p className="slide-content">
                    Now, let’s talk about <span className={"text-primaryGreen font-bold"}>solar energy production</span>. Solar panels
                    generate electricity when the sun is shining. That means they usually start producing in the morning, peak around noon (when sunlight is strongest), and produce nothing at night. On a clear day, the solar production curve looks like a big hump in the middle of the day (low in early morning, <i>highest at midday</i>, then low again by sunset). Solar is clean and renewable – but obviously, it only works with sunlight!
                </p>

                <LineGraph
                    data={data}
                    width={700}
                    height={300}
                    xLabel={energyProductionProfile.data[0].x_label}
                    yLabel={energyProductionProfile.data[0].y_label}
                    title={energyProductionProfile.title}
                    xTicks={24}
                    colors={colors}
                />

                <button className="nextSlideButton" onClick={goToNext}>What when energy consumption and production don't match?</button>
            </div>
        </div>
    );
};

export default SolarPower;
