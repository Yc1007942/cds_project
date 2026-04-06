import Phaser from "phaser";
import { AGENT_PERSONAS, type AgentPersona } from "../../../shared/agents";

// Event emitter for React <-> Phaser communication
export const gameEvents = new Phaser.Events.EventEmitter();

interface AgentSprite {
  persona: AgentPersona;
  container: Phaser.GameObjects.Container;
  body: Phaser.GameObjects.Graphics;
  eyes: Phaser.GameObjects.Graphics;
  nameText: Phaser.GameObjects.Text;
  statusText: Phaser.GameObjects.Text;
  thinkingDots: Phaser.GameObjects.Text;
  homeX: number;
  homeY: number;
  state: "idle" | "walking" | "thinking" | "talking" | "done";
  bobOffset: number;
}

interface UserSprite {
  container: Phaser.GameObjects.Container;
  body: Phaser.GameObjects.Graphics;
  glow: Phaser.GameObjects.Graphics;
  nameText: Phaser.GameObjects.Text;
  postBubble: Phaser.GameObjects.Container | null;
}

export class SimulationScene extends Phaser.Scene {
  private agents: AgentSprite[] = [];
  private userSprite: UserSprite | null = null;
  private particles: Phaser.GameObjects.Graphics[] = [];
  private gridGraphics: Phaser.GameObjects.Graphics | null = null;
  private time_counter = 0;
  private isSimulating = false;

  constructor() {
    super({ key: "SimulationScene" });
  }

  create() {
    const { width, height } = this.scale;

    // Draw subtle grid background
    this.gridGraphics = this.add.graphics();
    this.drawGrid(width, height);

    // Create user sprite in center
    this.createUserSprite(width / 2, height / 2);

    // Create agent sprites in a circular layout around the center
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2.8;
    const positions = AGENT_PERSONAS.map((_, i) => {
      const angle = (i / AGENT_PERSONAS.length) * Math.PI * 2;
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });

    AGENT_PERSONAS.forEach((persona, i) => {
      this.createAgentSprite(persona, positions[i]!.x, positions[i]!.y);
    });

    // Listen for events from React
    gameEvents.on("start-simulation", this.startSimulation, this);
    gameEvents.on("agent-thinking", this.onAgentThinking, this);
    gameEvents.on("agent-responded", this.onAgentResponded, this);
    gameEvents.on("reset-simulation", this.resetSimulation, this);
    gameEvents.on("show-post", this.showPostBubble, this);

    // Handle resize
    this.scale.on("resize", (gameSize: Phaser.Structs.Size) => {
      this.handleResize(gameSize.width, gameSize.height);
    });
  }

  private drawGrid(width: number, height: number) {
    if (!this.gridGraphics) return;
    this.gridGraphics.clear();
    this.gridGraphics.lineStyle(1, 0x2a2a4a, 0.3);

    const spacing = 40;
    for (let x = 0; x < width; x += spacing) {
      this.gridGraphics.lineBetween(x, 0, x, height);
    }
    for (let y = 0; y < height; y += spacing) {
      this.gridGraphics.lineBetween(0, y, width, y);
    }
  }

  private createUserSprite(x: number, y: number) {
    const container = this.add.container(x, y);

    // Glow effect
    const glow = this.add.graphics();
    glow.fillStyle(0xa855f7, 0.15);
    glow.fillCircle(0, 0, 50);
    glow.fillStyle(0xa855f7, 0.08);
    glow.fillCircle(0, 0, 65);

    // Main body - larger circle
    const body = this.add.graphics();
    body.fillStyle(0xa855f7, 1);
    body.fillCircle(0, 0, 28);
    // Inner highlight
    body.fillStyle(0xc084fc, 0.6);
    body.fillCircle(-6, -8, 10);
    // Eyes
    body.fillStyle(0xffffff, 1);
    body.fillCircle(-8, -4, 5);
    body.fillCircle(8, -4, 5);
    body.fillStyle(0x1e1b4b, 1);
    body.fillCircle(-7, -3, 3);
    body.fillCircle(9, -3, 3);
    // Smile
    body.lineStyle(2, 0xffffff, 0.8);
    body.beginPath();
    body.arc(0, 2, 10, 0.2, Math.PI - 0.2, false);
    body.strokePath();

    const nameText = this.add.text(0, 40, "YOU", {
      fontSize: "14px",
      fontFamily: "Inter, sans-serif",
      fontStyle: "bold",
      color: "#c084fc",
      align: "center",
    });
    nameText.setOrigin(0.5);

    container.add([glow, body, nameText]);
    container.setDepth(10);

    this.userSprite = { container, body, glow, nameText, postBubble: null };
  }

  private createAgentSprite(
    persona: AgentPersona,
    x: number,
    y: number,
  ) {
    const container = this.add.container(x, y);

    // Main body
    const body = this.add.graphics();
    body.fillStyle(persona.color, 1);
    body.fillCircle(0, 0, 22);
    // Highlight
    const lighterColor = Phaser.Display.Color.IntegerToColor(persona.color);
    lighterColor.lighten(30);
    body.fillStyle(lighterColor.color, 0.5);
    body.fillCircle(-5, -6, 7);

    // Eyes
    const eyes = this.add.graphics();
    eyes.fillStyle(0xffffff, 1);
    eyes.fillCircle(-6, -3, 4);
    eyes.fillCircle(6, -3, 4);
    eyes.fillStyle(0x1a1a2e, 1);
    eyes.fillCircle(-5, -2, 2.5);
    eyes.fillCircle(7, -2, 2.5);

    const nameText = this.add.text(0, 32, persona.name.split(" ")[0], {
      fontSize: "11px",
      fontFamily: "Inter, sans-serif",
      fontStyle: "bold",
      color: persona.accentColor,
      align: "center",
    });
    nameText.setOrigin(0.5);

    const statusText = this.add.text(0, 46, "", {
      fontSize: "9px",
      fontFamily: "Inter, sans-serif",
      color: "#888",
      align: "center",
    });
    statusText.setOrigin(0.5);

    const thinkingDots = this.add.text(0, -35, "", {
      fontSize: "16px",
      fontFamily: "Inter, sans-serif",
      fontStyle: "bold",
      color: persona.accentColor,
      align: "center",
    });
    thinkingDots.setOrigin(0.5);

    container.add([body, eyes, nameText, statusText, thinkingDots]);
    container.setDepth(5);

    this.agents.push({
      persona,
      container,
      body,
      eyes,
      nameText,
      statusText,
      thinkingDots,
      homeX: x,
      homeY: y,
      state: "idle",
      bobOffset: Math.random() * Math.PI * 2,
    });
  }

  private showPostBubble(_data: { post: string }) {
    if (!this.userSprite) return;

    // Remove existing bubble
    if (this.userSprite.postBubble) {
      this.userSprite.postBubble.destroy();
    }

    const truncated =
      _data.post.length > 50 ? _data.post.substring(0, 47) + "..." : _data.post;

    const bubble = this.add.container(0, -55);

    const text = this.add.text(0, 0, truncated, {
      fontSize: "11px",
      fontFamily: "Inter, sans-serif",
      color: "#e2e8f0",
      align: "center",
      wordWrap: { width: 180 },
    });
    text.setOrigin(0.5);

    const padding = 10;
    const bg = this.add.graphics();
    bg.fillStyle(0x2d2b55, 0.95);
    bg.fillRoundedRect(
      -text.width / 2 - padding,
      -text.height / 2 - padding,
      text.width + padding * 2,
      text.height + padding * 2,
      8,
    );
    bg.lineStyle(1, 0xa855f7, 0.5);
    bg.strokeRoundedRect(
      -text.width / 2 - padding,
      -text.height / 2 - padding,
      text.width + padding * 2,
      text.height + padding * 2,
      8,
    );

    // Arrow pointing down
    bg.fillStyle(0x2d2b55, 0.95);
    bg.fillTriangle(-6, text.height / 2 + padding, 6, text.height / 2 + padding, 0, text.height / 2 + padding + 8);

    bubble.add([bg, text]);
    bubble.setDepth(20);
    this.userSprite.container.add(bubble);
    this.userSprite.postBubble = bubble;

    // Animate bubble appearing
    bubble.setScale(0);
    bubble.setAlpha(0);
    this.tweens.add({
      targets: bubble,
      scaleX: 1,
      scaleY: 1,
      alpha: 1,
      duration: 300,
      ease: "Back.easeOut",
    });
  }

  private startSimulation() {
    this.isSimulating = true;
    const { width, height } = this.scale;
    const centerX = width / 2;
    const centerY = height / 2;

    // Move agents toward center with staggered timing
    this.agents.forEach((agent, i) => {
      const angle = (i / this.agents.length) * Math.PI * 2 - Math.PI / 2;
      const radius = 120;
      const targetX = centerX + Math.cos(angle) * radius;
      const targetY = centerY + Math.sin(angle) * radius;

      agent.state = "walking";
      agent.statusText.setText("approaching...");

      this.tweens.add({
        targets: agent.container,
        x: targetX,
        y: targetY,
        duration: 800 + i * 300,
        ease: "Quad.easeInOut",
        delay: i * 200,
        onComplete: () => {
          agent.state = "idle";
          agent.statusText.setText("listening");
          gameEvents.emit("agent-arrived", { agentId: agent.persona.id });
        },
      });
    });
  }

  private onAgentThinking(data: { agentId: string }) {
    const agent = this.agents.find((a) => a.persona.id === data.agentId);
    if (!agent) return;

    agent.state = "thinking";
    agent.statusText.setText("thinking...");

    // Add pulsing glow
    this.tweens.add({
      targets: agent.container,
      scaleX: 1.1,
      scaleY: 1.1,
      duration: 400,
      yoyo: true,
      repeat: -1,
      ease: "Sine.easeInOut",
    });
  }

  private onAgentResponded(data: { agentId: string; response: string }) {
    const agent = this.agents.find((a) => a.persona.id === data.agentId);
    if (!agent) return;

    // Stop thinking animation
    this.tweens.killTweensOf(agent.container);
    agent.container.setScale(1);

    agent.state = "done";
    agent.statusText.setText("responded");
    agent.thinkingDots.setText("");

    // Brief flash effect
    const flash = this.add.graphics();
    flash.fillStyle(agent.persona.color, 0.3);
    flash.fillCircle(0, 0, 35);
    agent.container.add(flash);

    this.tweens.add({
      targets: flash,
      alpha: 0,
      duration: 500,
      onComplete: () => flash.destroy(),
    });

    // Show brief speech indicator
    const speechBubble = this.createMiniSpeechBubble(agent);
    this.tweens.add({
      targets: speechBubble,
      alpha: 0,
      y: speechBubble.y - 20,
      duration: 2000,
      delay: 1500,
      onComplete: () => speechBubble.destroy(),
    });
  }

  private createMiniSpeechBubble(agent: AgentSprite): Phaser.GameObjects.Container {
    const bubble = this.add.container(0, -40);

    const bg = this.add.graphics();
    bg.fillStyle(Phaser.Display.Color.IntegerToColor(agent.persona.color).color, 0.2);
    bg.fillRoundedRect(-12, -8, 24, 16, 6);
    bg.lineStyle(1, agent.persona.color, 0.5);
    bg.strokeRoundedRect(-12, -8, 24, 16, 6);

    const dots = this.add.text(0, 0, "...", {
      fontSize: "12px",
      fontFamily: "Inter, sans-serif",
      fontStyle: "bold",
      color: agent.persona.accentColor,
    });
    dots.setOrigin(0.5);

    bubble.add([bg, dots]);
    bubble.setDepth(15);
    agent.container.add(bubble);

    return bubble;
  }

  private resetSimulation() {
    this.isSimulating = false;

    // Remove post bubble
    if (this.userSprite?.postBubble) {
      this.userSprite.postBubble.destroy();
      this.userSprite.postBubble = null;
    }

    // Move agents back to home positions
    this.agents.forEach((agent) => {
      this.tweens.killTweensOf(agent.container);
      agent.container.setScale(1);

      this.tweens.add({
        targets: agent.container,
        x: agent.homeX,
        y: agent.homeY,
        duration: 600,
        ease: "Quad.easeInOut",
      });

      agent.state = "idle";
      agent.statusText.setText("");
      agent.thinkingDots.setText("");
    });
  }

  private handleResize(width: number, height: number) {
    this.drawGrid(width, height);

    // Reposition user sprite
    if (this.userSprite) {
      this.userSprite.container.setPosition(width / 2, height / 2);
    }

    // Reposition agent home positions
    const positions = [
      { x: 100, y: 100 },
      { x: width - 100, y: 100 },
      { x: 100, y: height - 100 },
      { x: width - 100, y: height - 100 },
    ];

    this.agents.forEach((agent, i) => {
      agent.homeX = positions[i].x;
      agent.homeY = positions[i].y;
      if (!this.isSimulating) {
        agent.container.setPosition(positions[i].x, positions[i].y);
      }
    });
  }

  update(_time: number, delta: number) {
    this.time_counter += delta;

    // Idle bobbing animation for all sprites
    this.agents.forEach((agent) => {
      const bobSpeed = agent.state === "thinking" ? 0.008 : 0.003;
      const bobAmount = agent.state === "thinking" ? 4 : 2;
      const bob =
        Math.sin(this.time_counter * bobSpeed + agent.bobOffset) * bobAmount;
      agent.body.setY(bob);
      agent.eyes.setY(bob);

      // Thinking dots animation
      if (agent.state === "thinking") {
        const dotCount = Math.floor((this.time_counter / 400) % 4);
        agent.thinkingDots.setText(".".repeat(dotCount));
      }

      // Eye tracking toward center when simulating
      if (this.isSimulating && this.userSprite) {
        const dx = this.userSprite.container.x - agent.container.x;
        const lookDir = dx > 0 ? 1 : -1;
        agent.eyes.setX(lookDir * 1.5);
      } else {
        agent.eyes.setX(0);
      }
    });

    // User sprite glow pulse
    if (this.userSprite) {
      const pulse = 0.9 + Math.sin(this.time_counter * 0.003) * 0.1;
      this.userSprite.glow.setScale(pulse);
    }
  }
}
