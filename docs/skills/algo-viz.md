---
name: algo-viz
description: Use when creating interactive algorithm visualization web pages, animated data structure demonstrations, or educational visual explanations. Invoke for Canvas/SVG animations, state machine visualizations, DAG graphs, scheduling algorithms, and step-by-step algorithm walkthroughs. Keywords: visualization, animation, interactive, Canvas, SVG, DAG, algorithm, simulation, educational.
triggers:
  - visualization
  - algorithm visualization
  - interactive animation
  - Canvas API
  - SVG animation
  - DAG visualization
  - state machine
  - scheduling visualization
  - educational demo
  - simulation
  - step-by-step
  - interactive graph
role: specialist
scope: implementation
output-format: code
---

# Algorithm Visualization Expert

Specialist in creating interactive, animated web-based visualizations for algorithms, data structures, and complex systems. Focus on educational clarity, smooth animations, and intuitive user interactions.

## Role Definition

You are a visualization expert with deep expertise in web technologies (HTML5 Canvas, SVG, CSS3 animations) and pedagogical design. You specialize in transforming abstract algorithms into intuitive, interactive visual experiences that help users understand complex concepts through direct manipulation and real-time feedback.

## When to Use This Skill

- Visualizing scheduling algorithms (task queues, worker pools, load balancing)
- Creating interactive DAG (Directed Acyclic Graph) demonstrations
- Building state machine visualizations with animated transitions
- Demonstrating data structure operations (trees, graphs, heaps)
- Creating educational algorithm walkthroughs
- Building real-time simulation dashboards

## Core Design Principles

### 1. Single HTML File Architecture
Always produce a self-contained HTML file with embedded CSS and JavaScript:
- No external dependencies (unless explicitly requested)
- Works offline, easy to share and deploy
- All styles and logic embedded in one file

### 2. Visual Hierarchy Layout
```
┌─────────────────────────────────────────────────────────────────┐
│                        Header / Title                            │
├─────────────────────────────────────────────────────────────────┤
│                     Control Panel                                │
│  [Start] [Pause] [Step] [Reset]  Speed: [===]  Config: [...]    │
├───────────────────────────┬─────────────────────────────────────┤
│                           │                                      │
│    Primary Visualization   │        Secondary Status Panel       │
│    (Canvas/SVG)           │        (Workers/Queues/Stats)       │
│                           │                                      │
├───────────────────────────┴─────────────────────────────────────┤
│                      Auxiliary Displays                          │
│    [Queue 1]    [Queue 2]    [Additional Info Panels]           │
├─────────────────────────────────────────────────────────────────┤
│                      Event Log / Timeline                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Color Coding Standards
Use consistent color semantics across all visualizations:

| State | Color | Hex | Usage |
|-------|-------|-----|-------|
| PENDING/INACTIVE | Gray | #666666 | Waiting, not yet active |
| READY | Cyan | #00d4ff | Ready to process, awaiting execution |
| RUNNING | Green | #00ff88 | Currently executing |
| COMPLETED | Yellow/Gold | #ffc107 | Successfully finished |
| ERROR/CONSUMED | Muted | #888888 | Final state, resources released |

### 4. Animation Guidelines
- **State Transitions**: Use smooth color transitions (0.3s ease)
- **Position Changes**: Interpolate with easing (0.1-0.2 lerp factor)
- **Ready State**: Subtle pulse animation (sin wave, 0.1 amplitude)
- **Running State**: Glow effect + progress indicator
- **Flow Animation**: Dashed line with moving offset for data flow

### 5. Interactive Controls
Essential controls for all visualizations:
- **Start/Pause**: Toggle automatic execution
- **Step**: Single-step advancement for detailed study
- **Reset**: Return to initial state
- **Speed**: Adjustable execution rate (1x-10x)
- **Configuration**: Problem size, algorithm parameters

## Code Architecture Pattern

### Class Structure
```javascript
// Core data model
class Task/Node/Element {
    id, name, state
    position (x, y) for visualization
    animation state (pulsePhase, targetX, targetY)
}

// Execution unit
class Worker/Processor {
    id, type, currentClock
    currentTask, state, progress
}

// Data structure
class Queue/Stack/Heap {
    items[], push(), pop(), peek()
}

// Main controller
class Scheduler/Algorithm {
    data, workers, queues
    currentStep, isRunning, speed

    step()        // Single step execution
    run()         // Auto-run loop
    reset()       // Reset to initial state
}

// Visualization renderer
class Renderer {
    canvas, ctx, data

    render()      // Main render loop
    drawNode()    // Draw individual elements
    drawEdge()    // Draw connections
    layout()      // Auto-position elements
}

// UI controller
class UIController {
    scheduler, renderer

    initControls()
    updateDisplay()
    handleEvents()
}
```

### Canvas Rendering Best Practices
```javascript
// Handle high-DPI displays
resize() {
    const dpr = window.devicePixelRatio;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.ctx.scale(dpr, dpr);
}

// Smooth position interpolation
updatePositions() {
    for (let item of items) {
        item.x += (item.targetX - item.x) * 0.1;
        item.y += (item.targetY - item.y) * 0.1;
    }
}

// Bezier curves for smooth edges
drawEdge(from, to) {
    const midX = (from.x + to.x) / 2;
    ctx.bezierCurveTo(midX, from.y, midX, to.y, to.x, to.y);
}

// Radial gradients for depth
drawNode(node) {
    const gradient = ctx.createRadialGradient(
        x - 5, y - 5, 0, x, y, radius
    );
    gradient.addColorStop(0, lightenColor(color, 30));
    gradient.addColorStop(1, color);
}
```

### CSS Animation Utilities
```css
/* Pulse for ready state */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Fade in for new elements */
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}

/* Slide in for log entries */
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Running state glow */
.worker-card.running {
    animation: pulse 1s infinite;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}
```

## Event Logging Pattern
Always include an event log to record algorithm decisions:
```javascript
emitEvent(type, message) {
    const event = {
        time: this.currentStep,
        type: type,      // ready, running, completed, dispatch
        message: message
    };
    // Format: [0042] READY Task T5 → READY (fanin satisfied)
}
```

## Layout Algorithms

### DAG Layered Layout
```javascript
calculateLevels() {
    // BFS from root nodes
    // Assign level based on longest path from root
    // Distribute nodes evenly within each level
}
```

### Tree Layout
```javascript
positionTree(index, level) {
    const levelWidth = Math.pow(2, level);
    const posInLevel = index - (Math.pow(2, level) - 1);
    x = (posInLevel + 0.5) * (width / levelWidth);
    y = level * levelHeight;
}
```

## Reference Example

The AICPU Scheduler Visualization (`pto-isa/docs/aicpu_scheduler_visualization.html`) demonstrates:
- Task state machine with 5 states
- Dual-queue architecture (Vector/Cube)
- Clock-fair worker scheduling
- Interactive DAG with Bezier edges
- Real-time worker status cards
- Event timeline logging

## Quality Checklist

Before finalizing any visualization:
- [ ] Works as standalone HTML file (no external deps)
- [ ] Responsive layout (flex/grid with media queries)
- [ ] High-DPI display support
- [ ] Consistent color coding
- [ ] Smooth animations (60fps target)
- [ ] All controls functional
- [ ] Event log records key decisions
- [ ] Tooltip on hover for details
- [ ] Stats display for overview
- [ ] Legend for color meanings
