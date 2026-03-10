import PixiSplash from "../PixiSplash.jsx";

const IntroductionSlide = ({ goToNext }) => {
  return (
      <div>
        {/* Pixi background */}
        <PixiSplash numRings={4} />

        {/* Foreground content */}
        <div className="absolute inset-0 flex flex-col items-center
                            justify-center text-center z-10">
          <div className="max-w-md text-center">
            <h1 className="slide-title">Welcome to the Energy Community Demo</h1>
            <p className="slide-content">
              Hello and welcome! 👋
              <br/><br/>

              Today, we’ll explore <strong>energy communities</strong>, neighborhoods or groups that produce and share
              energy together. This interactive demo will walk you through the basics of how we use energy, how solar
              power works, and why sometimes there’s too much or too little at the wrong times. Don’t worry if you’re not
              an expert, we’ll keep things friendly and fun. By the end, you’ll even get to <strong>build your own energy
              community</strong> and see how sharing energy can benefit everyone. Ready to dive in?
            </p>

            <button
              onClick={goToNext}
              className="nextSlideButton"
            >
              Let’s get started! →
            </button>
          </div>
      </div>
    </div>
  );
};

export default IntroductionSlide;
