import { useEffect, useState } from 'react';
import BatteryEffectGraph from '../BatteryEffectGraph';
import PixiSplash from "../PixiSplash.jsx"; // added import

const BatteryInteractive = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryBlue = getCSSVariable('--color-primaryBlue');
    const primaryGreen = getCSSVariable('--color-primaryGreen');

    return (
        <div>
            <PixiSplash numRings={3} />

            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="slide-title">How a Battery Helps (Try it Out)</h1>
                <div className="slide-content">
                    A battery can help bridge the mismatch between when solar energy is produced and when it’s needed. <br/>

                    Use the slider below to adjust the battery size:
                    <br />
                    <ul className="text-left list-disc ml-6">
                        <li>🌞 At midday, solar often produces more than needed. This is shown by the green shaded area — energy being stored in the battery.</li>
                        <li>🌙 After sunset, stored energy can be used to reduce grid usage. This is shown by the blue shaded area — energy being discharged.</li>
                    </ul>
                    <br/>

                    As the battery size increases, you’ll see more midday surplus being stored, and more evening demand being met without the grid.
                    <br/>

                    <BatteryEffectGraph />
                </div>


                <button className="nextSlideButton" onClick={goToNext}>Let's weigh the pros and cons →</button>
            </div>
        </div>
    );
};

export default BatteryInteractive;
