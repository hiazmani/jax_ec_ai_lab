import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyConsumptionProfile from '../../data/energy_consumption_ex.json';
import PixiSplash from "../PixiSplash.jsx";

const DailyEnergyUse = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryBlue = getCSSVariable('--color-primaryBlue');
    const colors = [primaryBlue];

    useEffect(() => {
        const lineData = energyConsumptionProfile.data.map(line => ({
            label: line.legend_title,
            color: primaryBlue,
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
                <h1 className="slide-title">Our Daily Energy Use</h1>
                <p className="slide-content">
                    Every day, we all use electricity for things like lights, refrigerators, heating, or charging phones. This <span className={"text-primaryBlue font-bold"}>energy consumption</span> isn’t
                    the same all the time – it changes over the day. For example, many households use little electricity late at night, a bit when people get up (coffee makers, toasters), then often less during mid-day if everyone’s out, and a lot in the evening (cooking dinner, TV, etc.). Factories or schools have their own patterns (often high use during working hours, low at night). Understanding these patterns is important: it tells us <i>when</i> we need power the most.
                </p>

                <LineGraph
                    data={data}
                    width={700}
                    height={300}
                    xLabel={energyConsumptionProfile.data[0].x_label}
                    yLabel={energyConsumptionProfile.data[0].y_label}
                    title={energyConsumptionProfile.title}
                    colors={colors}
                    xScaleType="linear"
                    xTicks={24} // Add this line to specify the number of ticks
                    activeIndices={[0]}
                />

                <button className="nextSlideButton" onClick={goToNext}>Let's look at energy production →</button>
            </div>
        </div>

    );
};

export default DailyEnergyUse;
