import { motion } from 'framer-motion';
import { useState } from 'react';
import aiLabLogo from '../assets/ai_lab_logo.webp';
import './SlideIndicator.css';

const SlideIndicator = ({ slides, activeSlide, onSlideChange }) => {
    const [hovered, setHovered] = useState(null);

    return (
        <div className="slide-indicator-bar">
            <div className="left-space" />

            <div className="slide-indicator">
                <div className="dots-wrapper">
                    {slides.map((slide, idx) => (
                        <div
                            key={idx}
                            className={`indicator-dot ${idx === activeSlide ? 'active' : ''}`}
                            onClick={() => onSlideChange(idx)}
                            onMouseEnter={() => setHovered(idx)}
                            onMouseLeave={() => setHovered(null)}
                        >
                            <motion.div
                                className="tooltip"
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: hovered === idx ? 1 : 0, y: hovered === idx ? 0 : -10 }}
                                transition={{ duration: 0.2 }}
                            >
                                {slide.label}
                            </motion.div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="logo-container">
                <img src={aiLabLogo} alt="AI Lab logo" className="ai-lab-logo" />
            </div>
        </div>
    );
};

export default SlideIndicator;
