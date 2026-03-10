import { useEffect, useState } from 'react';
import PixiSplash from "../PixiSplash.jsx"; // added import

const BatterySolution = ({ goToNext }) => {
    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={3} />
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center z-10">
                <h1 className="slide-title mb-6">Solution 1: Storing Energy in Batteries</h1>

                <div className="slide-content">
                    {/* Left column: text */}
                    <p className="slide-content">
                        One way to handle the mismatch is using <i>batteries</i>. A battery can <i>store extra energy</i> when you don’t need it and <i>release it later</i> when you do. For example, imagine your solar panels
                        produce a lot at midday – more than your house is using. Instead of wasting that, a battery can charge
                        up with the surplus. Then in the evening, when you need electricity but the sun is down, the battery
                        can discharge that stored energy to power your lights or TV. In essence, it’s like saving energy for
                        a rainy (or dark) time!
                    </p>

                    {/* Right column: image */}
                    <div className="w-1/3 slide-content">
                        <img
                            src="battery_solution.png"
                            alt="Illustration of solar energy charging a battery and powering a house at night"
                            className="max-w-full h-auto"
                        />
                    </div>
                </div>

                <button className="nextSlideButton" onClick={goToNext}>
                    How does a battery help out? →
                </button>
            </div>
        </div>
    );
};

export default BatterySolution;
