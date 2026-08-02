# I built an AI-powered workspace safety & posture monitor (100% Local / In-browser)

Hey devs,

I wanted to share a side project I’ve been working on to solve a common problem for anyone spending hours at a desk: bad posture and messy workspaces. 

It's a web app built for productivity, ergonomics, and desk safety.

Try the Web App (No setup required): https://ergoandpostureai.vercel.app/

---

## Features

### 1. Desk Safety & Reach Zone Assessment
- Hazard Detection: Detects risky items (like water cups near tech) or distractions and alerts you to move them out of your reach zone.
- Custom Reach Zone: Calculates your safe workspace boundary based on your arm span, suggesting where to place essential tools.

### 2. Focus & Posture Monitor Mode
- Smart Pomodoro Timer: Keeps you productive during deep work sessions.
- AI Posture Tracking: Detects slouching, side leaning, and sitting too close to the screen.
- Voice Alerts: Gives instant voice warnings if bad posture persists for over 3 seconds.
- One-click Calibration: Snaps a quick reference photo before each session to set your personal "ideal posture" baseline. (Tip: If you adjust your webcam or seat height, simply re-calibrate!)

---

## Privacy First

Webcam access is sensitive, so privacy was the main priority:
- Zero video or image data is ever sent to any server.
- All AI processing runs 100% locally in your web browser.
- Your camera stream never leaves your machine.

---

## Tech Stack

- Frontend: HTML5, CSS3, JavaScript
- AI / Computer Vision: YOLO

---

I’d love to hear your thoughts, feedback, or any feature suggestions! Feel free to check out the repo or leave a comment below.
