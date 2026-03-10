import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx"; // added import

const RenewableEnergyPuzzle = ({ goToNext }) => {
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
            <PixiSplash numRings={3} /> {/* added PixiSplash */}

            {/* Foreground content */}

            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">The Renewable Energy Puzzle</h1>
                <p className="slide-content">
                    So we have a puzzle: <i>sometimes we have extra solar energy</i> that nobody is using at that moment, and other times we need energy but solar isn’t available. If nothing is done, that extra midday solar can go to waste (or be sent back to the grid for very little reward), and evening shortages mean buying electricity (often from non-renewable sources) from the grid.
                </p>

                <img src="/energy_puzzle.png" alt="Energy Puzzle" style={{maxWidth: '40%', height: 'auto'}}/>

                <p className="slide-content">
                    This isn’t ideal for making the most of our clean energy. How can we fix this mismatch so that <i>less green energy is wasted</i> and <i>everyone still gets the power they need</i>? Let’s explore two key solutions: <i>batteries</i> and <i>energy sharing</i>.
                </p>

                <button className="nextSlideButton" onClick={goToNext}>Let's look at possible solutions...→</button>
            </div>
        </div>
    );
};

export default RenewableEnergyPuzzle;
