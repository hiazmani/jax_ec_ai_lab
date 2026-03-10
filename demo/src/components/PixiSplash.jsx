import React, { useState, useEffect } from 'react';
import { Application } from '@pixi/react';
import { Splash } from './Splash';

export default function PixiSplash({ numRings = 4 }) {
    const [width, setWidth] = useState(window.innerWidth);
    const [height, setHeight] = useState(window.innerHeight);

    useEffect(() => {
        const handleResize = () => {
            setWidth(window.innerWidth);
            setHeight(window.innerHeight);
        };

        window.addEventListener('resize', handleResize);

        // Clean up the event listener when the component unmounts
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        // TODO: Fix the height based on the actual height of the indicator instead of hardcoding 80.
        <Application width={width} height={height} backgroundAlpha={0}>
            <Splash width={width} height={height} numRings={numRings} />
        </Application>
    );
}
