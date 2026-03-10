const ExplanationSandboxSlide = ({ goToNext }) => {
    return (
        <div className="slide explanation-sandbox-slide slide-container">
            <h1 className="slide-title">Explanation Sandbox</h1>
            <p className="slide-content">
                Welcome to the Energy Sandbox! Here, you can experiment with different scenarios to see how households interact within a community. First, you’ll learn about the different types of agents that control each home, the assets they have access to, and the objectives they aim to achieve.

                Think of an agent as the decision-maker for a household. It decides when to buy, sell or store energy. Agents can differ in three important ways:
                <ul>
                    <li>Sophistication: Agents can have different decision-making processes</li>
                    <li>Assets: Houses can have access to different assets</li>
                    <li>Objectives: Not alla gents aim for the same goal. By mixing different objectives, you can see how households with diverse motivations interact in the community.</li>
                </ul>
            </p>

            <button
                className={"nextSlideButton mt-6"}
                onClick={goToNext}>Let's dive in... →</button>
        </div>
    );
};

export default ExplanationSandboxSlide;
