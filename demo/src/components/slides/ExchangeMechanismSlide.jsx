const ExchangeMechanismSlide = ({ goToNext }) => {
    return (
        <div className="slide explanation-sandbox-slide slide-container">
            <h1 className="slide-title">Exchange Mechanisms</h1>
            <p className="slide-content">
                Now that you’ve assembled an energy community with different agents and objectives, the next step is to
                decide how these households trade energy among themselves. We currently offer two types of exchange
                mechanisms, each of which shapes the flow of energy in unique ways.
            </p>

            <p className="slide-content">
                <strong><b>Auction Market</b></strong>: In a double auction, each agent sets a price at which it is
                willing to buy or sell electricity. The market then matches these offers, determining who trades with
                whom and at what price. This approach is familiar from traditional marketplaces and is known for its
                transparency and fairness. By choosing this option, you’ll see how households interact when they
                directly set their own prices and let a central clearing process find the best matches.
            </p>

            <p className="slide-content">
                <strong><b>RL-Based Price Planner</b></strong>: In contrast, a reinforcement learning (RL) price planner
                means that instead of each household bidding its own price, a specialized AI agent sets a single, dynamic
                price for the entire community. This agent continuously adapts the price based on current supply and
                demand, seeking to maintain balance and efficiency. With this mechanism, households make decisions about
                when to buy or sell in response to a price that’s always adjusting to local conditions.
            </p>

            <p className="slide-content">
                As you switch between these two mechanisms and run simulations, watch how the community’s behavior changes.
                Does the double auction lead to more stable outcomes, or can the adaptive RL pricing scheme achieve a better balance?
                Exploring these differences can help you understand the trade-offs between various market designs and guide discussions
                about which rules might best serve a given energy community.
            </p>

            <button className="nextSlideButton mt-6" onClick={goToNext}>Let's try these in your community... →</button>
        </div>
    );
};

export default ExchangeMechanismSlide;
