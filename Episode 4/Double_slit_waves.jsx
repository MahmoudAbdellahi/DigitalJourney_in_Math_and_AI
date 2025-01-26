import React, { useState, useEffect } from "react";

const WaveInterference = () => {
  const width = 1200;
  const height = 600;
  const slitY = height * 0.5;
  const slitSpacing = 120;
  const slitWidth = 45;
  const sourceX = width * 0.2;
  const sourceY = height * 0.5;
  const detectorX = width * 0.8;
  const slitUpperPosition = slitY - slitSpacing / 2;
  const slitLowerPosition = slitY + slitSpacing / 2;
  const wavelength = 40;

  const [waves, setWaves] = useState([]);
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [accumulatedPattern, setAccumulatedPattern] = useState([]);

  const calculateWaveFromSlit = (x, y, slitY, time, phase) => {
    const dx = x - width * 0.4;
    const dy = y - slitY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const spread = Math.min(1, dx / 100);
    return Math.cos((2 * Math.PI * distance) / wavelength - time * 0.1 + phase) * Math.exp((-dy * dy) / (1000 * (1 + spread)));
  };

  const calculateInterference = (x, y, time, phase) => {
    if (x <= width * 0.4) return 0;
    const wave1 = calculateWaveFromSlit(x, y, slitUpperPosition, time, phase);
    const wave2 = calculateWaveFromSlit(x, y, slitLowerPosition, time, phase);
    return wave1 + wave2;
  };

  const generateInterferencePoints = () => {
    if (!isRunning || waves.length === 0) return [];

    const points = [];
    const spacing = 8;

    waves.forEach((wave) => {
      if (wave.x > width * 0.4) {
        const startX = width * 0.4;
        const endX = wave.x;

        for (let x = startX; x < endX; x += spacing) {
          for (let y = slitY - 150; y <= slitY + 150; y += spacing) {
            const amplitude = calculateInterference(x, y, time, wave.phase);
            if (Math.abs(amplitude) > 0.1) {
              points.push({ x, y, intensity: amplitude });
            }
          }
        }
      }
    });

    return points;
  };

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTime((t) => t + 1);

      if (time % 20 === 0) {
        setWaves((prev) => [
          ...prev.slice(-5),
          {
            id: Math.random(),
            x: sourceX,
            phase: time * 0.1,
          },
        ]);
      }

      setWaves((prev) =>
        prev
          .map((wave) => {
            const newX = wave.x + 6;
            if (newX >= width) return null;
            return { ...wave, x: newX };
          })
          .filter(Boolean)
      );
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning, time]);

  return (
    <div className="flex flex-col items-center p-4 bg-gray-900">
      <button
        onClick={() => setIsRunning(!isRunning)}
        className="mb-4 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
      >
        {isRunning ? "Pause" : "Start"} Experiment
      </button>

      <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ width: `${width}px`, height: `${height}px` }}>
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px)",
            backgroundSize: "30px 30px",
          }}
        />

        <div
          className="absolute"
          style={{
            left: sourceX - 30,
            top: sourceY - 30,
            width: "60px",
            height: "60px",
            background: "radial-gradient(circle, rgba(255,255,255,0.9) 0%, rgba(255,215,0,0.8) 40%, rgba(255,215,0,0) 70%)",
            borderRadius: "50%",
            boxShadow: "0 0 30px #FFD700, 0 0 60px #FFD700, 0 0 90px rgba(255,215,0,0.5)",
          }}
        />

        {waves.map((wave) => {
          const waveHeight = 120;
          return (
            <div key={wave.id}>
              <div
                className="absolute"
                style={{
                  left: wave.x,
                  top: sourceY - waveHeight / 2,
                  width: "4px",
                  height: `${waveHeight}px`,
                  background: "linear-gradient(to bottom, transparent, rgba(255, 215, 0, 0.6), transparent)",
                  boxShadow: "0 0 20px rgba(255, 215, 0, 0.4)",
                }}
              />
              <div
                className="absolute"
                style={{
                  left: wave.x - 15,
                  top: sourceY - waveHeight / 2 - 15,
                  width: "30px",
                  height: `${waveHeight + 30}px`,
                  background: "radial-gradient(ellipse at center, rgba(255,215,0,0.2) 0%, transparent 70%)",
                }}
              />
            </div>
          );
        })}

        {generateInterferencePoints().map((point, i) => {
          const brightness = Math.abs(point.intensity);
          return (
            <div
              key={i}
              className="absolute rounded-full"
              style={{
                left: point.x - 2,
                top: point.y - 2,
                width: "4px",
                height: "4px",
                background: `rgba(255, 215, 0, ${brightness * 0.6})`,
                boxShadow: `0 0 ${brightness * 12}px rgba(255, 215, 0, ${brightness * 0.4})`,
                opacity: 0.8,
              }}
            />
          );
        })}

        <div
          className="absolute bg-blue-500"
          style={{
            left: width * 0.4,
            top: 0,
            width: "6px",
            height: slitY - slitSpacing / 2 - slitWidth / 2,
            boxShadow: "0 0 12px #3B82F6, 0 0 24px rgba(59, 130, 246, 0.5)",
          }}
        />
        <div
          className="absolute bg-blue-500"
          style={{
            left: width * 0.4,
            top: slitY - slitSpacing / 2 + slitWidth / 2,
            width: "6px",
            height: slitSpacing - slitWidth,
            boxShadow: "0 0 12px #3B82F6, 0 0 24px rgba(59, 130, 246, 0.5)",
          }}
        />
        <div
          className="absolute bg-blue-500"
          style={{
            left: width * 0.4,
            top: slitY + slitSpacing / 2 + slitWidth / 2,
            width: "6px",
            height: height - (slitY + slitSpacing / 2 + slitWidth / 2),
            boxShadow: "0 0 12px #3B82F6, 0 0 24px rgba(59, 130, 246, 0.5)",
          }}
        />

        <div
          className="absolute bg-indigo-500 opacity-30"
          style={{
            left: detectorX,
            top: 0,
            width: "3px",
            height: "100%",
            boxShadow: "0 0 12px #6366F1",
          }}
        />
      </div>
    </div>
  );
};

export default WaveInterference;
