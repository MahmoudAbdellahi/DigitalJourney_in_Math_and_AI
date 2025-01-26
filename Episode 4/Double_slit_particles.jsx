import React, { useState, useEffect } from "react";

const DoubleSlit = () => {
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

  const waveSpeed = 6;
  const [waves, setWaves] = useState([]);
  const [particles, setParticles] = useState([]);
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [detectionPoints, setDetectionPoints] = useState([]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTime((t) => t + 1);

      if (time % 20 === 0) {
        setWaves((prev) => [
          ...prev,
          {
            id: Math.random(),
            x: sourceX,
            y: sourceY,
            phase: Math.random() * Math.PI * 2,
            measured: false,
          },
        ]);
      }

      setWaves((prev) =>
        prev
          .map((wave) => {
            const newX = wave.x + waveSpeed;

            if (newX >= width * 0.4 && !wave.measured) {
              const numParticles = 2;
              for (let i = 0; i < numParticles; i++) {
                const targetY = i === 0 ? slitUpperPosition : slitLowerPosition;
                setParticles((particles) => [
                  ...particles,
                  {
                    id: Math.random(),
                    x: newX,
                    y: targetY + (Math.random() - 0.5) * slitWidth,
                    vx: waveSpeed,
                    vy: 0,
                  },
                ]);
              }
              return null;
            }

            return {
              ...wave,
              x: newX,
            };
          })
          .filter(Boolean)
      );

      setParticles((prev) =>
        prev
          .map((particle) => {
            const newX = particle.x + particle.vx;

            if (newX >= detectorX) {
              setDetectionPoints((points) => [...points, particle.y]);
              return null;
            }

            return {
              ...particle,
              x: newX,
              y: particle.y + particle.vy,
            };
          })
          .filter(Boolean)
      );
    }, 20);

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
          const distanceFromSource = wave.x - sourceX;
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

        {particles.map((particle) => (
          <div
            key={particle.id}
            className="absolute"
            style={{
              left: particle.x - 9,
              top: particle.y - 9,
              width: "18px",
              height: "18px",
              background: "radial-gradient(circle, rgba(255,255,255,0.9) 0%, rgba(255,215,0,0.8) 50%, rgba(255,215,0,0) 100%)",
              borderRadius: "50%",
              boxShadow: "0 0 15px #FFD700, 0 0 30px rgba(255,215,0,0.5)",
            }}
          />
        ))}

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
          className="absolute"
          style={{
            left: width * 0.39,
            top: 0,
            width: "6px",
            height: "100%",
            background: "linear-gradient(90deg, rgba(144, 238, 144, 0.2) 0%, rgba(144, 238, 144, 0.4) 50%, rgba(144, 238, 144, 0.2) 100%)",
            boxShadow: "0 0 15px rgba(144, 238, 144, 0.5)",
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

        {detectionPoints.map((y, i) => (
          <div
            key={i}
            className="absolute w-4 h-4 rounded-full"
            style={{
              left: detectorX - 6,
              top: y - 6,
              background: "radial-gradient(circle, rgba(255,255,255,0.9) 0%, rgba(255,215,0,0.8) 50%, rgba(255,215,0,0) 100%)",
              boxShadow: "0 0 15px #FFD700, 0 0 30px rgba(255,215,0,0.5)",
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default DoubleSlit;
