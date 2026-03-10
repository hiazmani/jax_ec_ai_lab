const EnergyCommunitySlide = ({ goToNext }) => {
    return (
        <div className="slide energy-community-slide slide-container">
            <h1 className="slide-title">Energy Community</h1>
            <p className="slide-content">
                In practice, multiple households with unique energy consumption and production patterns join together to form an energy community. When one home produces more solar power than it can use, it can share that surplus with neighbors who need it. This local exchange reduces waste, lowers reliance on the broader grid, and often leads to more affordable electricity. By working together, communities can better handle variations in production and consumption, creating a more flexible, resilient, and cost-effective energy system. Let's see how this works in practice.
            </p>

            <button className="nextSlideButton" onClick={goToNext}>Let's create our own... →</button>
        </div>
    );
};

export default EnergyCommunitySlide;
