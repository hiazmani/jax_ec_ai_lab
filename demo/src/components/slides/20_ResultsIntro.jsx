import { useEffect, useState } from 'react';
import PixiSplash from '../PixiSplash.jsx';

const ResultsIntro = ({ goToNext }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Fake loading bar over 2 seconds (20 increments of 100ms)
    const totalDuration = 2000;
    const stepTime = 100;
    const steps = totalDuration / stepTime;
    let current = 0;
    const interval = setInterval(() => {
      current += 1;
      setProgress(Math.min((current / steps) * 100, 100));
      if (current >= steps) {
        clearInterval(interval);
      }
    }, stepTime);
    return () => clearInterval(interval);
  }, []);

  const isLoaded = progress >= 100;

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Centered content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6 text-center">
        <h1 className="slide-title mb-4">Results: Energy Sharing and Costs</h1>
        <p className="slide-content mb-6 max-w-2xl">
          You just simulated <span className="text-red-600">10 days</span> in the life of your energy community. Let’s explore what happened — how much energy was used, how much was saved, and how fair and sustainable your choices were.
        </p>

        {/* Fake loading bar */}
        <div className="w-full max-w-md bg-gray-300 rounded-full h-4 overflow-hidden mb-6">
          <div
            className="h-full bg-blue-600 transition-[width] duration-100 ease-linear"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-gray-700 mb-4">
          {isLoaded ? 'Results are ready!' : `Loading results... ${Math.round(progress)}%`}
        </p>

        <button
          onClick={goToNext}
          disabled={!isLoaded}
          className={`px-6 py-2 rounded shadow text-white ${
            isLoaded ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-400 cursor-not-allowed'
          }`}
        >
          {isLoaded ? "Let's see the economic results!" : 'Loading...'}
        </button>
      </div>
    </div>
  );
};

export default ResultsIntro;
