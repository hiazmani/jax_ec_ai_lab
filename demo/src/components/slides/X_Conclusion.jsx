import React, { useState } from 'react';
import { motion } from 'framer-motion';
import PixiSplash from '../PixiSplash.jsx';

const Conclusion = ({ goToNext, goToHome }) => {
  const learnItems = [
    "The challenge of matching solar production with usage — and how to solve it.",
    "The trade-offs between buying a battery vs. joining a community.",
    "How rules and AI strategies can shape energy outcomes.",
    "That no solution fits everyone — smart design matters.",
    "That communities can make us greener and more resilient.",
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={2} className="absolute inset-0 z-0" />

      {/* Foreground content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center text-center px-8 py-6">
        <motion.h1
          className="slide-title text-3xl md:text-4xl mb-4"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Conclusion: Empowering Communities for a Green Future 🌱
        </motion.h1>

        <motion.p
          className="slide-content max-w-3xl text-base md:text-lg leading-relaxed text-gray-700 mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          We hope this demo has been illuminating! You’ve seen how energy flows in a community, and how
          smart use of <strong>batteries</strong> and <strong>sharing</strong> turns a group of homes into a cooperative mini-grid.
          <br /><br />
          <strong>Energy communities</strong> empower people to take control of their energy, save money, and use more renewables.
          This is not a future concept—it’s happening now.
        </motion.p>

        <motion.div
          className="rounded-lg bg-green-100 bg-opacity-50 border-2 border-green-700 p-8 mb-8"
          initial="hidden"
          animate="visible"
          variants={{ visible: { transition: { staggerChildren: 0.15 } }, hidden: {} }}
        >
          <motion.h2
            className="text-xl md:text-2xl mb-4 text-green-700"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            Here’s what you learned:
          </motion.h2>
          <motion.ul
            className="list-none space-y-4 text-left max-w-2xl text-sm md:text-base text-gray-800"
            initial="hidden"
            animate="visible"
            variants={{ visible: { transition: { staggerChildren: 0.15 } }, hidden: {} }}
          >
            {learnItems.map((text, i) => (
              <motion.li
                key={i}
                variants={{ hidden: { opacity: 0, x: 30 }, visible: { opacity: 1, x: 0 } }}
                className="flex items-start"
              >
                <span className="mr-2 text-green-600">✅</span>
                <span>{text}</span>
              </motion.li>
            ))}
          </motion.ul>
        </motion.div>

        {/* QR Code and Credits */}
        <div className="flex flex-col items-center mb-8">
          <br/>
          <div className="p-4 rounded shadow-md mb-2">
            {/* Placeholder QR code; replace src with actual URL when available */}
            <img src="/assets/qr.png" alt="Demo QR Code" className="w-32 h-32" />
          </div>
          <p className="text-xs text-gray-600">
            Scan to access the full demo, code, and research paper online.
          </p>
        </div>

        {/* Final buttons */}
        <motion.div
          className="flex space-x-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.5 }}
        >
          <button
            onClick={goToHome}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
          >
            Replay or Explore Again →
          </button>
          <button
            onClick={() => alert('Thank you!')}
            className="bg-gray-200 text-gray-800 px-6 py-2 rounded hover:bg-gray-300"
          >
            Exit
          </button>
        </motion.div>

        {/* Credits */}
        <br/>
        <p className="mt-6 text-xs text-gray-500 max-w-3xl">
          Demo inspired by <a href="https://ncase.me/" className="underline">Nicky Case</a>.
          <br/>
          Funded by Innoviris and the Flanders AI Program.
        </p>
      </div>
    </div>
  );
};

export default Conclusion;
