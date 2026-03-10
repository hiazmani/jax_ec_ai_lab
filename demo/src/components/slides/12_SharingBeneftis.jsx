import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx";
import {motion} from "framer-motion";

const SharingBenefits = ({ goToNext }) => {
    const [data, setData] = useState([]);

    // You can still use colors for later LineGraph or theming
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryGreen = getCSSVariable('--color-primaryGreen');

    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} />

            {/* Main Content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center z-10">
                <h1 className="slide-title">Benefits of Sharing Locally</h1>

                <p className="slide-content mb-6">
                    Why is energy sharing great? Here are some benefits:
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
                                <li><strong>Less wasted solar:</strong> Instead of throwing away excess noon solar,
                                    someone else
                                    uses it. More renewable energy gets consumed.
                                </li>
                                <li><strong>Lower bills for everyone:</strong> Neighbors can buy excess solar power
                                    often
                                    cheaper than from the grid, and sellers earn a bit – so both sides save money.
                                </li>
                                <li><strong>No one left out:</strong> Even a household with no solar panels can directly
                                    get
                                    clean energy from a neighbor’s panel. This helps renters and apartment dwellers.
                                </li>
                                <li><strong>Community resilience:</strong> People sharing energy can help each other
                                    during peak
                                    times or outages – like a shared water well.
                                </li>
                                <li><strong>Smaller batteries needed:</strong> With sharing, not everyone needs their
                                    own
                                    battery. Energy can flow where it's needed, when it's needed.
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
                                <li><strong>Complexity:</strong> Requires planning and coordination. Who shares with whom and when?</li>
                                <li><strong>Privacy concerns:</strong> Sharing energy data might raise privacy issues. Who sees what?</li>
                                <li><strong>Regulatory hurdles:</strong> Some places have strict rules about energy sharing.</li>
                            </ul>
                        </div>
                    </motion.div>
                </div>

                {/* Spacer */}
                <div className="mt-6"/>

                <button className="nextSlideButton" onClick={goToNext}>
                    Storing or Sharing?
                </button>
            </div>
        </div>
    );
};

export default SharingBenefits;
