import { useEffect, useState } from 'react';
import CommunityBuilderContainer from "../CommunityBuilderContainer.jsx";

const SharingInteractive = ({ goToNext }) => {
    const [data, setData] = useState([]);
    // Function to get CSS variable value
    const getCSSVariable = (variable) => {
        return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    };

    const primaryBlue = getCSSVariable('--color-primaryBlue');
    const primaryGreen = getCSSVariable('--color-primaryGreen');

    return (
        <div className="slide energy-profiles-slide slide-container" style={{ width: '100%', padding: '40px', boxSizing: 'border-box' }}>
            <h1 className="slide-title">Try it Out: Sharing in an Energy Community</h1>

            <CommunityBuilderContainer />

            <button className="nextSlideButton" onClick={goToNext}>Let's weigh the pros and cons →</button>
        </div>
    );
};

export default SharingInteractive;
