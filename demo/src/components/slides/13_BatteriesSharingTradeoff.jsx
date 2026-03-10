import { useEffect, useState } from 'react';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx";

const BatteriesSharingTradeoff = ({ goToNext }) => {
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

            {/* Main Content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center z-10">
                <h1 className="slide-title">Sharing vs. Batteries – What’s the Trade-off?  </h1>
                <div className="slide-content">
                    So we've seen two solutions: batteries and sharing. But how do they stack up against each other? <br />
                    <ul className="text-left list-disc ml-6">
                        <li><strong>Batteries</strong>: Great for independence – you store your own energy. However, you bear the full cost, and a battery only serves your home (unless it’s a <i>shared community battery</i>. It’s like each person having their own fridge for leftovers.</li>
                        <li><strong>Sharing</strong>: Great for collaboration – excess energy goes to neighbors, potentially cutting costs for all. It requires coordination and trust (or a good system to track who gives and gets).</li>
                    </ul>
                    <br />
                    In reality, the best approach might be a <strong>mix</strong>: some storage plus sharing. For instance, a community could have a <i>shared battery</i> that everyone uses, and they trade energy too. The key idea is using <i>local renewable energy</i> as efficiently as possible, whether by storing it or moving it around to where it’s needed.
                </div>

                {/* Add image */}
                <div className="w-1/4 slide-content">
                    <img
                        src="battery_vs_sharing.png"
                        alt="Illustration of battery vs sharing."
                        className="max-w-full h-auto"
                    />
                </div>

                <button className="nextSlideButton" onClick={goToNext}>Let's now build our own energy community!</button>
            </div>
        </div>
    );
};

export default BatteriesSharingTradeoff;
