import PixiSplash from "../PixiSplash";

const HomeSlide = ({ goToNext }) => {
    return (
        <div>
            {/* Pixi background */}
            <PixiSplash numRings={4} />

            {/* Foreground content */}
            <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
                <h1 className="text-4xl font-bold text-brown">
                  AI-DRIVEN ENERGY COMMUNITIES
                </h1>
                <h2 className="text-lg mt-2 text-brown">
                 A sandbox for AI-Driven Energy Communities
                </h2>
                <button
                    className="nextSlideButton mt-6"
                    onClick={goToNext}
                >
                    START →
                </button>
            </div>
        </div>
    );
};

export default HomeSlide;
