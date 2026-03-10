import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx";
import { motion } from 'framer-motion';

const BatteriesProsAndCons = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryBlue = getCSSVariable('--color-primaryBlue');
    const primaryGreen = getCSSVariable('--color-primaryGreen');

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} />

            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">Batteries: Pros and Cons</h1>
                <p className="slide-content mb-6">
                    Batteries can be amazing, but they have <span className={"text-green-500 font-bold"}>upsides</span> and <span className={"text-red-500 font-bold"}>downsides</span>:
                </p>
                <div className="flex flex-col justify-center items-start gap-6 w-full max-w-3xl mx-auto mb-8">
                    {/* Pros box */}
                    <motion.div
                        initial={{opacity: 0, x: -30}}
                        animate={{opacity: 1, x: 0}}
                        transition={{duration: 0.6, delay: 0.2}}
                        className="bg-green-100/60 border border-green-300 rounded-lg p-4 w-full shadow-sm px-6"
                    >
                        <div className="bg-green-100/60 border border-green-300 rounded-lg p-4 w-full shadow-sm px-6">
                            <h2 className="text-green-800 font-semibold text-2xl mb-2 text-center">✅ Pros</h2>
                            <ul className="list-disc list-inside text-left text-sm text-green-900 space-y-2">
                                <li><strong>More self-sufficiency:</strong> You use more of your own solar energy
                                    instead of
                                    relying on the grid, even at night.
                                </li>
                                <li><strong>Backup power:</strong> During outages, a battery can keep the lights on (at
                                    least
                                    for a while).
                                </li>
                            </ul>
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{opacity: 0, x: 30}}
                        animate={{opacity: 1, x: 0}}
                        transition={{duration: 0.6, delay: 0.4}}
                        className="bg-red-100/60 border border-red-300 rounded-lg p-4 w-full shadow-sm px-6"
                    >
                        {/* Cons box */}
                        <div className="bg-red-100/60 border border-red-300 rounded-lg p-4 w-full shadow-sm px-6">
                            <h2 className="text-red-800 font-semibold text-2xl mb-2 text-center">❌ Cons</h2>
                            <ul className="list-disc list-inside text-left text-sm text-red-900 space-y-2">
                                <li><strong>Cost:</strong> Home batteries are still quite expensive. A battery that can
                                    hold
                                    enough for a whole night can cost many thousands of euros.
                                </li>
                                <li><strong>Limited capacity:</strong> A battery has a finite size. If there’s a long
                                    cloudy
                                    spell or a very long usage period, it might run out. Once full, extra solar energy
                                    can’t be
                                    stored.
                                </li>
                                <li><strong>Lifespan:</strong> Batteries don’t last forever; after years and charge
                                    cycles, they
                                    hold less energy and might need replacement.
                                </li>
                            </ul>
                        </div>
                    </motion.div>
                </div>

                {/* Right column: image */}
                <div className="w-1/9 slide-content">
                    <img
                        src="battery_solution.png"
                        alt="Illustration of solar energy charging a battery and powering a house at night"
                        className="max-w-full h-auto"
                    />
                </div>

                <button className="nextSlideButton" onClick={goToNext}>How about energy sharing?→</button>
            </div>
        </div>
    );
};

export default BatteriesProsAndCons;
