import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx"; // added import

const ConsumptionProductionMismatch = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryBlue = getCSSVariable('--color-primaryBlue');
    const primaryGreen = getCSSVariable('--color-primaryGreen');
    const colors = [primaryGreen, primaryBlue];

    useEffect(() => {
        const lineData = energyProfile.data.map(line => ({
            label: line.legend_title,
            color: line.color,
            points: line.x.map((x, i) => ({ x: x, y: line.y[i] })),
        }));
        setData(lineData);
    }, []);

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} /> {/* added PixiSplash */}

            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">When Use and Production Don’t Match</h1>

                {/* Callout box */}
                <div
                    className="bg-yellow-100/80 border-l-4 border-yellow-400 text-yellow-900 text-sm p-4 max-w-4xl rounded mb-6 mt-10 shadow-sm">
                    <p className="font-semibold text-lg">
                        ⚡ Our <span className="text-primaryBlue font-bold">energy consumption</span> often doesn’t line
                        up with <span className="text-primaryGreen font-bold">solar energy production</span>.
                    </p>
                </div>
                <br/>

                <p className="slide-content max-w-2xl">
                    Often, the <i>demand</i> for power is highest in mornings and evenings, but <i>solar supply</i> is
                    highest at midday when demand can be lower. This mismatch means that sometimes there’s lots of solar
                    energy when we don’t need all of it, and later, when we do need power (evening), the sun isn’t
                    available. This type of curve is typically called a <a
                    href={"https://en.wikipedia.org/wiki/Duck_curve#:~:text=Without%20any%20form%20of%20energy,11"}>Duck
                    curve</a>. In other words, you might have <i>excess energy at noon</i> and a <i>shortage after
                    sunset</i>.
                </p>

                <LineGraph
                    data={data}
                    width={700}
                    height={300}
                    xLabel="Time (Hours)"
                    yLabel="Energy (kWh)"
                    title="Energy Profile"
                    colors={colors}
                    showAreaBetween={true}         // turn on shading
                    areaColor="#85965f"            // or any color
                    areaOpacity={0.15}             // adjust transparency
                />


                <button className="nextSlideButton" onClick={goToNext}>What affects energy consumption?</button>
            </div>
        </div>
    );
};

export default ConsumptionProductionMismatch;
