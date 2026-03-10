import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import solarProfile from '../../data/06_SunshineAndSeason/solar_production_profile.json';
import PixiSplash from "../PixiSplash.jsx";
import { motion } from 'framer-motion';


const buttonVariants = {
    hidden: { opacity: 0, x: 30 },
    visible: (i) => ({
        opacity: 1,
        x: 0,
        transition: { delay: i * 0.1, duration: 0.8, ease: 'easeOut' }
    })
};

const SunshineAndSeason = ({ goToNext }) => {
    // We'll store transformed data with "points" for each scenario.
    const [data, setData] = useState([]);
    // Active indices for toggling scenarios; default: all active.
    const [activeIndices, setActiveIndices] = useState([0]);

    useEffect(() => {
        // Transform the data: for each scenario, create a "points" array.
        const transformed = solarProfile.data.map(line => ({
            label: line.legend_title,
            color: line.color,
            points: line.x.map((x, i) => ({ x, y: line.y[i] }))
        }));
        setData(transformed);
    }, []);

    // Retrieve colors from the JSON directly, or use CSS variables as needed.
    const colors = data.map(d => d.color);

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} /> {/* added PixiSplash */}

            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">Sunshine and Season Matter</h1>
                <div className="slide-content">
                    Solar production isn’t the same every day. It can change with:
                    <ul className="text-left list-disc ml-6">
                        <li><strong>Weather</strong>: A cloudy day means your solar panels produce much less, with a flatter, lower peak. A clear sunny day yields a nice high peak at noon.</li>
                        <li><strong>Season</strong>: In winter, days are shorter and the sun is weaker, so even at noon you get less solar power (and maybe a narrower production window). In summer, long sunny days give you a big, broad production curve.</li>
                    </ul>
                    <br/>
                    Different scenarios lead to different production patterns:
                    <br/>

                    {/* Toggle buttons for solar production scenarios */}
                    <div className="flex space-x-4 my-4 justify-center">
                        <motion.div
                            className="flex space-x-4 my-4 justify-center"
                            initial="hidden"
                            animate="visible"
                        >
                            {data.map((scenario, i) => (
                                <motion.button
                                    key={i}
                                    custom={i}
                                    variants={buttonVariants}
                                    onClick={() => {
                                        setActiveIndices(prev =>
                                            prev.includes(i)
                                                ? prev.filter(idx => idx !== i)
                                                : [...prev, i]
                                        );
                                    }}
                                    className={`px-4 py-2 border rounded transition-colors ${
                                        activeIndices.includes(i)
                                            ? 'text-white'
                                            : 'text-gray-800'
                                    }`}
                                    style={{
                                        backgroundColor: activeIndices.includes(i) ? scenario.color : 'transparent',
                                        borderColor: scenario.color,
                                        borderWidth: '3px',
                                        margin: '0 4px'  // Add margin between buttons
                                    }}
                                >
                                    {scenario.label}
                                </motion.button>
                            ))}
                        </motion.div>
                    </div>
                </div>

                <LineGraph
                    data={data} // Pass full data for scale computation.
                    width={700}
                    height={300}
                    xLabel={solarProfile.x_label}
                    yLabel={solarProfile.y_label}
                    title={solarProfile.title}
                    colors={colors}
                    xTicks={10}
                    activeIndices={activeIndices.length > 0 ? activeIndices : [-1]}  // Only render lines for active scenarios, show none if no active
                />

                <button className="nextSlideButton" onClick={goToNext}>
                    The renewable energy puzzle →
                </button>
            </div>
        </div>
    );
};

export default SunshineAndSeason;
