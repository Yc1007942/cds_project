import { useEffect, useRef, useState } from "react";
import Phaser from "phaser";
import { SimulationScene } from "./SimulationScene";

interface PhaserGameProps {
  className?: string;
}

export default function PhaserGame({ className }: PhaserGameProps) {
  const gameRef = useRef<Phaser.Game | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  // Wait until the container has non-zero dimensions before creating Phaser
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Check if already sized
    if (el.clientWidth > 0 && el.clientHeight > 0) {
      setReady(true);
      return;
    }

    // Use ResizeObserver to wait for layout
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setReady(true);
          observer.disconnect();
        }
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Create the Phaser game only when container is ready
  useEffect(() => {
    if (!ready || !containerRef.current || gameRef.current) return;

    const w = containerRef.current.clientWidth;
    const h = containerRef.current.clientHeight;

    // Guard against zero dimensions (WebGL framebuffer needs > 0)
    if (w < 1 || h < 1) return;

    const config: Phaser.Types.Core.GameConfig = {
      type: Phaser.CANVAS, // Canvas renderer avoids WebGL framebuffer issues
      parent: containerRef.current,
      width: w,
      height: h,
      backgroundColor: "#0f0e1a",
      scene: [SimulationScene],
      scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
      render: {
        antialias: true,
        pixelArt: false,
        roundPixels: false,
      },
      audio: {
        noAudio: true,
      },
    };

    gameRef.current = new Phaser.Game(config);

    return () => {
      if (gameRef.current) {
        gameRef.current.destroy(true);
        gameRef.current = null;
      }
    };
  }, [ready]);

  return (
    <div
      ref={containerRef}
      className={`phaser-container ${className || ""}`}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
