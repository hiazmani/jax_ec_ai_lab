import { useEffect, useState } from 'react';
import LineGraph from '../LineGraph/index';
import energyProfile from '../../data/energy_profile_ex.json';
import PixiSplash from "../PixiSplash.jsx";

const SharingSolution = ({ goToNext }) => {
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
                <h1 className="slide-title">Solution 2 – Sharing in an Energy Community</h1>
                <p className="slide-content">
                    Now for the second solution: <i>energy communities</i> and sharing energy. Instead of each home acting alone, what if neighbors <i>team up</i>? In an <a href={"https://www.cet-power.com/en/news/energy-communities-shared-local-electricity-production/#:~:text=An%20energy%20community%20is%20a,be%20shared%2C%20exchanged%2C%20or%20sold))4"}>energy community</a>, people can <i>share, exchange, or even sell</i> electricity to each other local. Think of it like this: if your solar panels are making more power than you need right now, that extra can go to your neighbor who might be running their oven or charging a car. Later, if <i>you</i> need power, and maybe your neighbor has some spare (from their own solar or battery), they can send it your way. By coordinating, the community as a whole uses energy more efficiently. Essentially, your neighbors become your <i>energy backup</i>, and vice versa!
                </p>

                {/* Right column: image */}
                <div className="w-1/6 slide-content">
                    <img
                        src="energy_sharing_colored.png"
                        alt="Illustration of an energy community."
                        className="max-w-full h-auto"
                    />
                </div>

                <button className="nextSlideButton" onClick={goToNext}>What are the benefits of sharing locally?</button>
            </div>
        </div>
    );
};

export default SharingSolution;
