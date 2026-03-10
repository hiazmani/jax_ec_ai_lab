import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfiles from '../../data/05_DifferentLifestyles/energy_consumption_profiles.json';
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

const DifferentLifestyles = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Create toggle state with only the first profile active.
    const [activeIndices, setActiveIndices] = useState([0]);

    useEffect(() => {
        // For this slide, we assume each entry in energyProfiles.data corresponds to a profile.
        const transformedData = energyProfiles.data.map(line => ({
            ...line,
            label: line.legend_title,
            color: line.color,
            points: line.x.map((x, i) => ({ x: x, y: line.y[i] }))
        }));
        setData(transformedData);
    }, []);

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} /> {/* added PixiSplash */}
            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">Different Lifestyles, Different Patterns</h1>
                <div className="slide-content">
                    To understand the mismatch between energy production and consumption, we need to look at what people do with energy. Different lifestyles and activities use energy in different ways. This means that the <i>energy consumption</i> curve can look very different depending on what people are doing.
                    <ul className="text-left list-disc ml-6">
                        <li><strong>9-to-5 Worker</strong>: Away during the day, so low usage from 9am–5pm and higher usage in morning and evening.</li>
                        <li><strong>Work-from-Home</strong>: Steady usage throughout the day (computer, kettle, etc.), and still some in the evening.</li>
                        <li><strong>School Building</strong>: High usage during school hours (say 8am–3pm) on weekdays, almost none at night or weekends.</li>
                    </ul>
                    <br/>
                    Let’s look at a few typical consumption profiles:
                    <br/>

                    {/* Render dynamic toggle buttons */}
                    <div className="flex space-x-4 my-4 justify-center">
                        <motion.div
                            className="flex space-x-4 my-4 justify-center"
                            initial="hidden"
                            animate="visible"
                        >
                            {data.map((profile, i) => (
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
                                        backgroundColor: activeIndices.includes(i) ? profile.color : 'transparent',
                                        borderColor: profile.color,
                                        borderWidth: '3px',
                                        margin: '0 4px'
                                    }}
                                >
                                    {profile.label}
                                </motion.button>
                            ))}
                        </motion.div>
                    </div>
                </div>

                <LineGraph
                    data={data}  // Pass full data to compute scales
                    width={700}
                    height={300}
                    xLabel={energyProfiles.x_label}
                    yLabel={energyProfiles.y_label}
                    title={energyProfiles.title}
                    colors={data.map(d => d.color)}  // colors from JSON
                    xTicks={24}
                    activeIndices={activeIndices.length > 0 ? activeIndices : [-1]}  // Only draw active lines, show none if no active
                />

                <button className="nextSlideButton" onClick={goToNext}>
                    What affects energy production?
                </button>
            </div>
        </div>

    );
};

export default DifferentLifestyles;
