const sectionDesk = document.getElementById('desk_assessment_section');
const sectionWorkspace = document.getElementById('main_workspace_section');
const navStep1 = document.getElementById('nav_step1');
const navStep2 = document.getElementById('nav_step2');

const deskFileInput = document.getElementById('desk_file_input');
const deskResultImg = document.getElementById('desk_result_img');
const deskPlaceholder = document.getElementById('desk_placeholder');
const btnAnalyzeFrame = document.getElementById('btn_analyze_desk_frame');
const deskFeedbackText = document.getElementById('desk_feedback_text');
const deskScoreDisplay = document.getElementById('desk_score_display');
const deskStatusBadge = document.getElementById('desk_status_badge');
const recheckBanner = document.getElementById('recheck_banner');

const webcam = document.getElementById('webcam');
const webcamContainer = document.getElementById('webcam_container');
const btnCalibrate = document.getElementById('btn_calibrate');

const BACKEND_URL = "https://ergo-pomodoro-posture-ai.onrender.com";
const MODEL_PATH = "./yolov8n-pose.onnx";

const MODEL_SIZE = 256;
const inputFloat32Array = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);

let isStep1Completed = false;
let isMonitoring = false;
let sessionONNX = null;

let baselineEyeDist = 0;
let baselineShoulderY = 0;
let baselineShoulderWidth = 0;
let isCalibrated = false;

let smoothedEyeDist = 0;
let smoothedShoulderY = 0;
let smoothedShoulderWidth = 0;

let badPostureStartTime = 0;
let goodPostureStartTime = 0;
let isWarningActive = false;

const yoloCanvas = document.createElement('canvas');
yoloCanvas.width = MODEL_SIZE;
yoloCanvas.height = MODEL_SIZE;
const yoloCtx = yoloCanvas.getContext('2d', { willReadFrequently: true });

async function initYoloONNX() {
    if (sessionONNX) return true;

    const statusEl = document.getElementById('posture_status');
    try {
        if (statusEl) {
            statusEl.innerText = "LOADING AI MODEL...";
            statusEl.className = "font-bold text-amber-500 tracking-wide animate-pulse";
        }

        sessionONNX = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm']
        });

        if (statusEl) {
            statusEl.innerText = "READY TO CALIBRATE";
            statusEl.className = "font-bold text-primary tracking-wide";
        }
        return true;
    } catch (err) {
        if (statusEl) {
            statusEl.innerText = "MODEL NOT FOUND (404)";
            statusEl.className = "font-bold text-error tracking-wide";
        }
        alert("Lỗi: Không tìm thấy tệp 'yolov8n-pose.onnx'. Vui lòng tải tệp này và đặt vào cùng thư mục với index.html.");
        return false;
    }
}

function compressImage(file, maxWidth = 800, quality = 0.6) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                if (width > maxWidth) {
                    height = Math.round((height * maxWidth) / width);
                    width = maxWidth;
                }

                canvas.width = width;
                canvas.height = height;

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob(
                    (blob) => {
                        const compressedFile = new File([blob], file.name, {
                            type: 'image/jpeg',
                            lastModified: Date.now(),
                        });
                        resolve(compressedFile);
                    },
                    'image/jpeg',
                    quality
                );
            };
            img.onerror = (err) => reject(err);
        };
        reader.onerror = (err) => reject(err);
    });
}

function updateSidebar(activeStep) {
    if (activeStep === 1) {
        if (navStep1) navStep1.className = "px-4 py-3 text-primary font-bold bg-primary/10 rounded-xl font-label-caps text-label-caps translate-x-1 transition-transform block";
        if (navStep2) {
            if (isStep1Completed) {
                navStep2.className = "px-4 py-3 text-on-surface-variant font-medium hover:bg-surface-container-highest transition-all rounded-xl font-label-caps text-label-caps block";
                navStep2.innerText = "MONITOR & FOCUS";
            } else {
                navStep2.className = "px-4 py-3 text-on-surface-variant/50 font-medium cursor-not-allowed transition-all rounded-xl font-label-caps text-label-caps block";
                navStep2.innerText = "MONITOR & FOCUS 🔒";
            }
        }
    } else {
        if (navStep1) navStep1.className = "px-4 py-3 text-on-surface-variant font-medium hover:bg-surface-container-highest transition-all rounded-xl font-label-caps text-label-caps block";
        if (navStep2) {
            navStep2.className = "px-4 py-3 text-primary font-bold bg-primary/10 rounded-xl font-label-caps text-label-caps translate-x-1 transition-transform block";
            navStep2.innerText = "MONITOR & FOCUS";
        }
    }
}

window.showStep = function (stepNumber) {
    if (stepNumber === 1) {
        stopPostureAI();
        stopWebcam();
        if (sectionDesk) sectionDesk.classList.remove('hidden');
        if (sectionWorkspace) sectionWorkspace.classList.add('hidden');
        updateSidebar(1);
    } else {
        trySwitchStep2();
    }
};

window.trySwitchStep2 = function () {
    if (!isStep1Completed) {
        alert("Please complete Step 1 (upload an image for analysis) or click 'SKIP TO FOCUS WORKSPACE' to unlock!");
        return;
    }
    goToMainWorkspace();
};

window.goToMainWorkspace = async function () {
    isStep1Completed = true;
    if (sectionDesk) sectionDesk.classList.add('hidden');
    if (sectionWorkspace) sectionWorkspace.classList.remove('hidden');
    updateSidebar(2);

    const isCamReady = await startWebcam();
    if (isCamReady) {
        const isModelLoaded = await initYoloONNX();
        if (isModelLoaded) {
            startPostureAI();
        }
    }
};

if (deskFileInput) {
    deskFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (deskFeedbackText) deskFeedbackText.innerText = "Compressing & Analyzing desk setup with AI...";
        if (deskStatusBadge) {
            deskStatusBadge.className = "status-badge-warning mb-6";
            deskStatusBadge.innerText = "PROCESSING...";
        }

        try {
            const compressedFile = await compressImage(file, 800, 0.6);
            const reader = new FileReader();

            reader.onload = (event) => {
                if (deskResultImg) {
                    deskResultImg.src = event.target.result;
                    deskResultImg.classList.remove('hidden');
                }
                if (deskPlaceholder) deskPlaceholder.classList.add('hidden');
            };
            reader.readAsDataURL(compressedFile);

            const formData = new FormData();
            formData.append("file", compressedFile);
            formData.append("user_height", "170.0");
            formData.append("fatigue_level", "30");

            const response = await fetch(`${BACKEND_URL}/api/assess_desk`, {
                method: "POST",
                body: formData
            });

            const responseText = await response.text();

            if (!response.ok) {
                throw new Error(`Status ${response.status}: ${responseText}`);
            }

            let data;
            try {
                data = JSON.parse(responseText);
            } catch (err) {
                throw new Error("Lỗi JSON: " + responseText.substring(0, 100));
            }

            let scoreMatch = data.feedback ? data.feedback.match(/\*\*(\d+)\/100\*\*/) : null;
            let scoreVal = scoreMatch ? scoreMatch[1] : "78";

            if (deskScoreDisplay) {
                deskScoreDisplay.innerHTML = `${scoreVal}<span class="text-2xl text-on-surface-variant font-normal">/100</span>`;
            }

            if (deskStatusBadge) {
                deskStatusBadge.className = "status-badge-warning mb-6";
                deskStatusBadge.innerText = Number(scoreVal) >= 80 ? "OPTIMAL SETUP" : "NEEDS ADJUSTMENT";
            }

            if (deskFeedbackText) {
                let formattedFeedback = data.feedback
                    .replace(/### (.*?)\n/g, '<strong class="text-amber-500 text-lg">$1</strong><br/>')
                    .replace(/\* (.*?)\n/g, '- $1<br/>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                deskFeedbackText.innerHTML = formattedFeedback;
            }

            if (data.processed_image && deskResultImg) {
                deskResultImg.src = data.processed_image;
            }

            if (data.audio_base64 && data.audio_base64.length > 50) {
                const audioSrc = data.audio_base64.startsWith('data:') 
                    ? data.audio_base64 
                    : `data:audio/mp3;base64,${data.audio_base64}`;

                const audio = new Audio(audioSrc);
                audio.play().catch(() => {
                    const cleanText = removeMarkdownForSpeech(data.feedback);
                    playVoiceAlert(cleanText);
                });
            } else if (data.feedback) {
                const cleanText = removeMarkdownForSpeech(data.feedback);
                playVoiceAlert(cleanText);
            }

            if (recheckBanner) recheckBanner.classList.remove('hidden');
            isStep1Completed = true;
            updateSidebar(1);

        } catch (error) {
            if (deskFeedbackText) {
                deskFeedbackText.innerText = "Lỗi Server: " + error.message;
            }
            if (deskStatusBadge) {
                deskStatusBadge.className = "status-badge-error mb-6";
                deskStatusBadge.innerText = "CONNECTION ERROR";
            }
        }
    });
}

if (btnAnalyzeFrame) {
    btnAnalyzeFrame.addEventListener('click', () => {
        if (deskFileInput) {
            deskFileInput.click();
        }
    });
}

async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support camera access or requires HTTPS.");
        return false;
    }

    if (webcam && !webcam.srcObject) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } }
            });
            webcam.srcObject = stream;
            await new Promise((resolve) => {
                webcam.onloadedmetadata = () => resolve();
            });
            await webcam.play();
            return true;
        } catch (err) {
            const statusEl = document.getElementById('posture_status');
            if (statusEl) {
                statusEl.innerText = "CAMERA PERMISSION DENIED";
                statusEl.className = "font-bold text-error tracking-wide";
            }
            return false;
        }
    }
    return true;
}

function stopWebcam() {
    if (webcam && webcam.srcObject) {
        const tracks = webcam.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        webcam.srcObject = null;
    }
}

async function sendFrameToBackend(imageBase64) {
    const response = await fetch(`${BACKEND_URL}/api/analyze_frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_base64: imageBase64,
            is_calibration: false
        })
    });
    const data = await response.json();
}

function preprocessWebcamToTensor() {
    if (!webcam || webcam.readyState < 2 || webcam.videoWidth === 0) return null;

    yoloCtx.drawImage(webcam, 0, 0, MODEL_SIZE, MODEL_SIZE);
    const imageData = yoloCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const data = imageData.data;

    const area = MODEL_SIZE * MODEL_SIZE;

    for (let i = 0; i < area; i++) {
        inputFloat32Array[i]            = data[i * 4]     / 255.0; 
        inputFloat32Array[area + i]     = data[i * 4 + 1] / 255.0; 
        inputFloat32Array[2 * area + i] = data[i * 4 + 2] / 255.0; 
    }

    return new ort.Tensor('float32', inputFloat32Array, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function startPostureAI() {
    if (isMonitoring) return;
    isMonitoring = true;

    async function monitorLoop() {
        if (!isMonitoring) return;

        if (sessionONNX && webcam && webcam.readyState >= 2) {
            const inputTensor = preprocessWebcamToTensor();
            if (inputTensor) {
                try {
                    const inputName = sessionONNX.inputNames[0];
                    const feeds = {};
                    feeds[inputName] = inputTensor;

                    const results = await sessionONNX.run(feeds);
                    const outputName = sessionONNX.outputNames[0];
                    const outputTensor = results[outputName];

                    if (outputTensor && outputTensor.data) {
                        processYoloOutput(outputTensor);
                    }
                } catch (e) {}
            }
        }

        if (isMonitoring) {
            requestAnimationFrame(monitorLoop);
        }
    }

    monitorLoop();
}

function stopPostureAI() {
    isMonitoring = false;
}

function processYoloOutput(outputTensor) {
    if (!outputTensor || !outputTensor.data || !outputTensor.dims) return;

    const data = outputTensor.data;
    const dims = outputTensor.dims;

    let numChannels = 56;
    let numAnchors = 1344;
    let isChannelsFirst = true;

    if (dims.length === 3) {
        if (dims[1] > dims[2]) {
            numAnchors = dims[1];
            numChannels = dims[2];
            isChannelsFirst = false;
        } else {
            numChannels = dims[1];
            numAnchors = dims[2];
            isChannelsFirst = true;
        }
    }

    let maxScore = -1;
    let bestIdx = -1;

    for (let i = 0; i < numAnchors; i++) {
        const scoreIdx = isChannelsFirst ? (4 * numAnchors + i) : (i * numChannels + 4);
        const score = data[scoreIdx];

        if (score > maxScore) {
            maxScore = score;
            if (score > 0.05) { 
                bestIdx = i;
            }
        }
    }

    if (bestIdx === -1) return;

    const getKP = (kpIdx) => {
        const channelX = 5 + kpIdx * 3;
        const channelY = 5 + kpIdx * 3 + 1;
        const channelConf = 5 + kpIdx * 3 + 2;

        let x, y, conf;
        if (isChannelsFirst) {
            x = data[channelX * numAnchors + bestIdx];
            y = data[channelY * numAnchors + bestIdx];
            conf = data[channelConf * numAnchors + bestIdx];
        } else {
            x = data[bestIdx * numChannels + channelX];
            y = data[bestIdx * numChannels + channelY];
            conf = data[bestIdx * numChannels + channelConf];
        }
        return { x, y, conf };
    };

    const nose = getKP(0);
    const leftEye = getKP(1);
    const rightEye = getKP(2);
    const leftShoulder = getKP(5);
    const rightShoulder = getKP(6);

    const minConf = 0.03;

    let currEyeDist = 0;
    if (leftEye.conf >= minConf && rightEye.conf >= minConf) {
        currEyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
    } else if (nose.conf >= minConf && leftEye.conf >= minConf) {
        currEyeDist = Math.hypot(leftEye.x - nose.x, leftEye.y - nose.y) * 2;
    } else if (nose.conf >= minConf && rightEye.conf >= minConf) {
        currEyeDist = Math.hypot(rightEye.x - nose.x, rightEye.y - nose.y) * 2;
    }

    let currShoulderY = 0;
    let currShoulderWidth = 0;
    let shoulderTilt = 0;

    if (leftShoulder.conf >= minConf && rightShoulder.conf >= minConf) {
        currShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        currShoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
        shoulderTilt = Math.abs(leftShoulder.y - rightShoulder.y);
    } else if (leftShoulder.conf >= minConf) {
        currShoulderY = leftShoulder.y;
    } else if (rightShoulder.conf >= minConf) {
        currShoulderY = rightShoulder.y;
    } else if (nose.conf >= minConf) {
        currShoulderY = nose.y + 40; 
    }

    if (currEyeDist === 0 && currShoulderWidth > 0) {
        currEyeDist = currShoulderWidth / 2.5;
    } else if (currShoulderWidth === 0 && currEyeDist > 0) {
        currShoulderWidth = currEyeDist * 2.5;
    }

    if (currEyeDist === 0 && currShoulderWidth === 0) return;

    if (smoothedEyeDist === 0) {
        smoothedEyeDist = currEyeDist;
        smoothedShoulderY = currShoulderY;
        smoothedShoulderWidth = currShoulderWidth;
    } else {
        smoothedEyeDist = smoothedEyeDist * 0.75 + currEyeDist * 0.25;
        smoothedShoulderY = smoothedShoulderY * 0.75 + currShoulderY * 0.25;
        smoothedShoulderWidth = smoothedShoulderWidth * 0.75 + currShoulderWidth * 0.25;
    }

    const eyeEl = document.getElementById('metric_eye');
    const shoulderEl = document.getElementById('metric_shoulder');
    if (eyeEl) eyeEl.innerText = `${Math.round(smoothedEyeDist)} px`;
    if (shoulderEl) shoulderEl.innerText = `${Math.round(smoothedShoulderY)} px`;

    if (!isCalibrated) return;

    const currentRatio = smoothedEyeDist / (smoothedShoulderWidth || 1);
    const baselineRatio = baselineEyeDist / (baselineShoulderWidth || 1);

    let isBadPosture = false;
    let statusText = "GOOD POSTURE";

    if (currentRatio > baselineRatio * 1.2) {
        isBadPosture = true;
        statusText = "TOO CLOSE TO SCREEN";
    } else if (smoothedShoulderY > baselineShoulderY + 8) {
        isBadPosture = true;
        statusText = "SLOUCHING DETECTED";
    } else if (shoulderTilt > 15) {
        isBadPosture = true;
        statusText = "LEANING / SHOULDER TILTED";
    }

    handlePostureStatus(isBadPosture, statusText);
}

if (btnCalibrate) {
    btnCalibrate.addEventListener('click', async () => {
        const statusEl = document.getElementById('posture_status');

        const isCamReady = await startWebcam();
        if (!isCamReady) return;

        const modelReady = await initYoloONNX();
        if (!modelReady) return;

        startPostureAI();

        btnCalibrate.disabled = true;
        btnCalibrate.innerText = "Scanning (2.5s)...";
        if (statusEl) {
            statusEl.innerText = "SCANNING FOR PERSON...";
            statusEl.className = "font-bold text-amber-500 tracking-wide animate-pulse";
        }

        smoothedEyeDist = 0;

        await new Promise((resolve) => setTimeout(resolve, 2500));

        if (smoothedEyeDist > 0) {
            baselineEyeDist = smoothedEyeDist;
            baselineShoulderY = smoothedShoulderY;
            baselineShoulderWidth = smoothedShoulderWidth;

            isCalibrated = true;
            if (statusEl) {
                statusEl.innerText = "BASELINE SAVED";
                statusEl.className = "font-bold text-primary tracking-wide";
            }
            btnCalibrate.innerText = "Recalibrate";
        } else {
            if (statusEl) {
                statusEl.innerText = "NO PERSON DETECTED";
                statusEl.className = "font-bold text-error tracking-wide";
            }
            btnCalibrate.innerText = "Try Again";
        }

        btnCalibrate.disabled = false;
    });
}

function handlePostureStatus(isBadPosture, statusMessage) {
    const now = Date.now();
    const statusEl = document.getElementById('posture_status');

    if (isBadPosture) {
        goodPostureStartTime = 0;
        if (badPostureStartTime === 0) badPostureStartTime = now;

        const elapsedBadTime = now - badPostureStartTime;
        if (elapsedBadTime >= 3000) {
            isWarningActive = true;
            showWarningUI(statusMessage, statusEl);
        } else {
            const countdown = 3 - Math.floor(elapsedBadTime / 1000);
            if (statusEl) {
                statusEl.innerText = `${statusMessage} (${countdown}s)`;
                statusEl.className = "font-bold text-amber-500 tracking-wide";
            }
        }
    } else {
        if (goodPostureStartTime === 0) goodPostureStartTime = now;

        const elapsedGoodTime = now - goodPostureStartTime;
        if (elapsedGoodTime >= 1000) {
            badPostureStartTime = 0;
            if (isWarningActive) {
                hideWarningUI(statusEl);
                isWarningActive = false;
            } else if (statusEl) {
                statusEl.innerText = "GOOD POSTURE";
                statusEl.className = "font-bold text-primary tracking-wide";
            }
        }
    }
}

function showWarningUI(message, statusEl) {
    if (statusEl) {
        statusEl.innerText = `🚨 ${message} 🚨`;
        statusEl.className = "font-bold text-error tracking-wide animate-pulse";
    }
    if (webcamContainer) {
        webcamContainer.classList.add('ring-4', 'ring-red-500');
    }
    playVoiceAlert(message);
}

function hideWarningUI(statusEl) {
    if (statusEl) {
        statusEl.innerText = "GOOD POSTURE";
        statusEl.className = "font-bold text-primary tracking-wide";
    }
    if (webcamContainer) {
        webcamContainer.classList.remove('ring-4', 'ring-red-500');
    }
}

let timerInterval = null;
const TOTAL_SECONDS = 25 * 60;
let timeLeft = TOTAL_SECONDS;
let isRunning = false;
let completedSessions = 3;

const timerDisplay = document.getElementById('timer_display');
const timerProgress = document.getElementById('timer_progress');
const btnTimerStart = document.getElementById('btn_timer_start');
const btnTimerPause = document.getElementById('btn_timer_pause');
const btnTimerReset = document.getElementById('btn_timer_reset');
const sessionTracker = document.getElementById('session_tracker');

function updateTimerDisplay() {
    if (!timerDisplay) return;
    const m = Math.floor(timeLeft / 60).toString().padStart(2, '0');
    const s = (timeLeft % 60).toString().padStart(2, '0');
    timerDisplay.innerText = `${m}:${s}`;

    if (timerProgress) {
        const percentage = ((TOTAL_SECONDS - timeLeft) / TOTAL_SECONDS) * 100;
        timerProgress.style.width = `${percentage}%`;
    }
}

if (btnTimerStart) {
    btnTimerStart.addEventListener('click', () => {
        if (isRunning) return;
        isRunning = true;
        btnTimerStart.innerText = "Focusing...";

        timerInterval = setInterval(() => {
            if (timeLeft > 0) {
                timeLeft--;
                updateTimerDisplay();
            } else {
                clearInterval(timerInterval);
                isRunning = false;
                completedSessions++;
                if (sessionTracker) {
                    sessionTracker.innerText = `${completedSessions} of 4 Sessions Completed`;
                }
                btnTimerStart.innerText = "Start Session";
                alert("25 minutes completed! Time to stand up and stretch.");
                timeLeft = TOTAL_SECONDS;
                updateTimerDisplay();
            }
        }, 1000);
    });
}

if (btnTimerPause) {
    btnTimerPause.addEventListener('click', () => {
        clearInterval(timerInterval);
        isRunning = false;
        if (btnTimerStart) btnTimerStart.innerText = "Resume Session";
    });
}

if (btnTimerReset) {
    btnTimerReset.addEventListener('click', () => {
        clearInterval(timerInterval);
        isRunning = false;
        timeLeft = TOTAL_SECONDS;
        updateTimerDisplay();
        if (btnTimerStart) btnTimerStart.innerText = "Start Session";
    });
}

updateTimerDisplay();

(async function pingBackendToWakeUp() {
    try {
        await fetch(`${BACKEND_URL}/ping`, { method: 'GET' });
    } catch (e) {}
})();

document.addEventListener('click', () => {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.resume();
    }
}, { once: true });

let lastSpeechTime = 0;
let cachedVoice = null;

function initVoices() {
    if ('speechSynthesis' in window) {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            cachedVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google')) 
                       || voices.find(v => v.lang.startsWith('en')) 
                       || voices[0];
        }
    }
}

if ('speechSynthesis' in window) {
    initVoices();
    window.speechSynthesis.onvoiceschanged = initVoices;
}

function removeMarkdownForSpeech(text) {
    if (!text) return "";
    return text
        .replace(/#+/g, '')
        .replace(/\*+/g, '')
        .replace(/[\_\`\~\-\>🚨]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function playVoiceAlert(text) {
    const now = Date.now();
    if (now - lastSpeechTime < 5000) return; 
    lastSpeechTime = now;

    if (!('speechSynthesis' in window)) return;

    const cleanText = removeMarkdownForSpeech(text);
    if (!cleanText) return;

    window.speechSynthesis.cancel(); 

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    if (cachedVoice) {
        utterance.voice = cachedVoice;
    }

    window.speechSynthesis.resume();
    window.speechSynthesis.speak(utterance);
}
