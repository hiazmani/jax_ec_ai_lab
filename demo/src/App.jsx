import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import HomeSlide from './components/slides/0_HomeSlide';
import SlideIndicator from './components/SlideIndicator';
import './App.css';
import IntroductionSlide from "./components/slides/1_IntroductionSlide.jsx";
import aiLabLogo from "./assets/ai_lab_logo.webp";
import DailyEnergyUse from "./components/slides/2_DailyEnergyUse.jsx";
import SolarPower from "./components/slides/3_SolarPower.jsx";
import ConsumptionProductionMismatch from "./components/slides/4_ConsumptionProductionMismatch.jsx";
import DifferentLifestyles from "./components/slides/5_DifferentLifestyles.jsx";
import SunshineAndSeason from "./components/slides/6_SunshineAndSeason.jsx";
import RenewableEnergyPuzzle from "./components/slides/7_RenewableEnergyPuzzle.jsx";
import BatterySolution from "./components/slides/8_BatterySolution.jsx";
import BatteryInteractive from "./components/slides/9_BatteryInteractive.jsx";
import BatteriesProsAndCons from "./components/slides/10_BatteriesProsAndCons.jsx";
import SharingSolution from "./components/slides/11_SharingSolution.jsx";
import SharingBenefits from "./components/slides/12_SharingBeneftis.jsx";
import BatteriesSharingTradeoff from "./components/slides/13_BatteriesSharingTradeoff.jsx";
import BuildingCommunity from "./components/slides/14_BuildingCommunity.jsx";
import AddParticipant from "./components/slides/15_AddParticipants.jsx";
import EquipParticipants from "./components/slides/16_EquipParticipants.jsx";
import DecisionMakingParticipants from "./components/slides/17_DecisionMakingParticipants.jsx";
import EnergyMechanism from "./components/slides/18_EnergyMechanism.jsx";
import RunSimulation from "./components/slides/19_RunSimulation.jsx";
// Results slides
import ResultsIntro from "./components/slides/20_ResultsIntro.jsx";
import ResultsCost from "./components/slides/21_ResultsCost.jsx";
import ResultsFairness from "./components/slides/22_ResultsFairness.jsx";
import ResultsSufficiency from "./components/slides/23_ResultsSufficiency.jsx";
import ResultsEnvironment from "./components/slides/24_ResultsEnvironment.jsx";
import ResultsPeak from "./components/slides/25_ResultsPeakGrid.jsx";
import TryFairness from "./components/slides/30_TryScenarios.jsx";
// Conclusion
import X_Conclusion from "./components/slides/X_Conclusion.jsx";

import { SimulationConfigProvider } from './components/SimulationConfigContext';
import { ResultsProvider } from "./components/ResultContext.jsx";
import SharingInteractive from "./components/slides/11b_SharingInteractive.jsx";


function App() {
    const [activeSlide, setActiveSlide] = useState(0);
    const [direction, setDirection] = useState(0); // +1 for next, -1 for prev

    const goToNext = () => {
        if (activeSlide < slides.length - 1) {
            setDirection(1);
            setActiveSlide(activeSlide + 1);
        }
    };

    const goToPrev = () => {
        if (activeSlide > 0) {
            setDirection(-1);
            setActiveSlide(activeSlide - 1);
        }
    };

    const handleMissingData = () => {
        alert("This slide requires results from the simulation. Please run the simulation first.");
        setActiveSlide(19); // Redirect to the simulation slide
  }

    const slides = [
        { component: <HomeSlide goToNext={goToNext} />, label: "0. Home" },
        { component: <IntroductionSlide goToNext={goToNext} />, label: "1. Introduction" },
        { component: <DailyEnergyUse goToNext={goToNext} />, label: "2. Daily Energy Use" },
        { component: <SolarPower goToNext={goToNext} />, label: "3. Solar Power: Energy from the Sun" },
        { component: <ConsumptionProductionMismatch goToNext={goToNext} />, label: "4. When Use and Production Don't Match"},
        { component: <DifferentLifestyles goToNext={goToNext} />, label: "5. Different Lifestyles, Different Patterns"},
        { component: <SunshineAndSeason goToNext={goToNext} />, label: "6. Sunshine and Season Matter"},
        { component: <RenewableEnergyPuzzle goToNext={goToNext} />, label: "7. The Renewable Energy Puzzle"},
        { component: <BatterySolution goToNext={goToNext} />, label: "8. Solution 1: Batteries"},
        { component: <BatteryInteractive goToNext={goToNext} />, label: "9. Try it out: Batteries"},
        { component: <BatteriesProsAndCons goToNext={goToNext} />, label: "10. Batteries: Pros and Cons"},
        { component: <SharingSolution goToNext={goToNext} />, label: "11. Solution 2: Sharing in an Energy Community"},
        // { component: <SharingInteractive goToNext={goToNext} />, label: "11b. Try it out: Sharing"},
        { component: <SharingBenefits goToNext={goToNext} />, label: "12. Benefits of Sharing Locally"},
        { component: <BatteriesSharingTradeoff goToNext={goToNext} />, label: "13. Sharing vs. Batteries – What’s the Trade-off?"},
        { component: <BuildingCommunity goToScenarioCreation={goToNext} goToAgentSelection={
            () => {
                console.log("goToAgentSelection");
                setActiveSlide(17);
                setDirection(1);
            }
        } />, label: "14. Build Your Own Energy Community!"},
        { component: <AddParticipant goToNext={goToNext} />, label: "15. Add Participants"},
        { component: <EquipParticipants goToNext={goToNext} />, label: "16. Equip Participants"},
        { component: <DecisionMakingParticipants goToNext={goToNext} />, label: "17. Set Decision-Making"},
        { component: <EnergyMechanism goToNext={goToNext} />, label: "18. Set Energy Exchange"},
        { component: <RunSimulation goToNext={goToNext} />, label: "19. Run the Simulation"},
        // Results Pages
        { component: <ResultsIntro goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "20. Results"},
        { component: <ResultsCost goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "21. Results: Energy Costs"},
        { component: <ResultsFairness goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "22. Results: Fairness"},
        { component: <ResultsSufficiency goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "23. Results: Self-Sufficiency"},
        { component: <ResultsEnvironment goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "24. Results: Environment"},
        { component: <ResultsPeak goToNext={goToNext} handleMissingData={handleMissingData}/>, label: "25. Results: Peak Grid Load"},
        { component: <TryFairness goToNext={goToNext} />, label: "26. Try Different Scenarios"},
        { component: <X_Conclusion goToNext={goToNext} goToHome={() => setActiveSlide(0)} />, label: "Conclusion"},

        // { component: <ExplanationSandboxSlide goToNext={goToNext} />, label: "Explanation Sandbox" },
        // { component: <ExchangeMechanismSlide goToNext={goToNext} />, label: "Exchange Mechanisms" },
        // { component: <DataVisualizationSlide goToPrev={goToPrev} />, label: "Data Visualization" },
    ];

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowRight') {
                goToNext();
            } else if (e.key === 'ArrowLeft') {
                goToPrev();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeSlide]); // Make sure activeSlide is in the dependency list

    return (
        <ResultsProvider>
            <SimulationConfigProvider>
                <div className="App">
                    <div className="slide-wrapper" style={{width: '100%', overflow: 'hidden'}}>
                        <AnimatePresence mode="wait" custom={direction}>
                            <motion.div
                                key={activeSlide}
                                custom={direction}
                                initial={{ x: direction > 0 ? '100%' : '-100%', opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                exit={{ x: direction > 0 ? '-100%' : '100%', opacity: 0 }}
                                transition={{ duration: 0.4, ease: 'easeInOut' }}
                            >
                                {slides[activeSlide].component}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    <div className="slide-indicator-container w-screen">
                        <div className="slide-indicator-wrapper" style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                            <SlideIndicator
                                slides={slides}
                                activeSlide={activeSlide}
                                onSlideChange={setActiveSlide}
                            />
                            <img src={aiLabLogo} alt="AI Lab Logo" className="ai-lab-logo"/>
                        </div>
                    </div>
                </div>
            </SimulationConfigProvider>
        </ResultsProvider>
    );
}

export default App;
