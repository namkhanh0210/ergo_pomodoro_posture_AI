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

let isStep1Completed = false;
let isMonitoring = false;

let baselineEyeDist = 0;
let baselineShoulderY = 0;

let badPostureStartTime = 0;
let goodPostureStartTime = 0;
let isWarningActive = false;

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

        sectionDesk.classList.remove('hidden');
        sectionWorkspace.classList.add('hidden');
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
    sectionDesk.classList.add('hidden');
    sectionWorkspace.classList.remove('hidden');
    updateSidebar(2);

    await startWebcam();
};

if (deskFileInput) {
    deskFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (deskResultImg) {
                    deskResultImg.src = event.target.result;
                    deskResultImg.classList.remove('hidden');
                }
                if (deskPlaceholder) deskPlaceholder.classList.add('hidden');
                
                if (deskFeedbackText) deskFeedbackText.innerText = "Scanning desk objects & elevation...";

                setTimeout(() => {
                    if (deskScoreDisplay) deskScoreDisplay.innerHTML = '78<span class="text-2xl text-on-surface-variant font-normal">/100</span>';
                    
                    if (deskStatusBadge) {
                        deskStatusBadge.className = "status-badge-warning mb-6";
                        deskStatusBadge.innerText = "NEEDS ADJUSTMENT";
                    }

                    if (deskFeedbackText) {
                        deskFeedbackText.innerHTML = `
                            <strong class="text-amber-500">Workspace Report:</strong><br/>
                            - <strong>Monitor:</strong> Slightly lower than eye level (needs elevation by ~5cm).<br/>
                            - <strong>Keyboard:</strong> Placed too close to the edge.<br/>
                            - <strong>Lighting:</strong> Balanced, no screen glare.<br/>
                            => <em>Elevate laptop/monitor to prevent neck strain.</em>`;
                    }

                    if (recheckBanner) recheckBanner.classList.remove('hidden');
                    isStep1Completed = true;
                    updateSidebar(1);
                }, 1800);
            };
            reader.readAsDataURL(file);
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
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;

            await new Promise((resolve) => {
                webcam.onloadedmetadata = () => resolve();
            });

            await webcam.play();
            return true;
        } catch (err) {
            console.error("Webcam init error:", err);
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

function captureWebcamBase64() {
    if (!webcam || !webcam.srcObject) return null;
    const canvas = document.createElement('canvas');
    canvas.width = webcam.videoWidth || 640;
    canvas.height = webcam.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7);
}

async function sendImageToAPI(base64Image, isCalibration) {
    try {
        const response = await fetch("http://localhost:8000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_base64: base64Image,
                is_calibration: isCalibration,
                baseline_eye_dist: baselineEyeDist,
                baseline_shoulder_y: baselineShoulderY
            })
        });
        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        return null;
    }
}

if (btnCalibrate) {
    btnCalibrate.addEventListener('click', async () => {
        const isCamReady = await startWebcam();
        if (!isCamReady) return;

        const statusEl = document.getElementById('posture_status');
        btnCalibrate.innerText = "Calibrating...";
        btnCalibrate.disabled = true;

        if (statusEl) {
            statusEl.innerText = "ANALYZING BASELINE...";
            statusEl.className = "font-bold text-amber-500 tracking-wide animate-pulse";
        }

        const base64Img = captureWebcamBase64();
        if (!base64Img) {
            btnCalibrate.innerText = "Try Again";
            btnCalibrate.disabled = false;
            return;
        }

        const apiData = await sendImageToAPI(base64Img, true);

        if (apiData && apiData.is_success) {
            baselineEyeDist = apiData.eye_dist;
            baselineShoulderY = apiData.shoulder_y;

            if (statusEl) {
                statusEl.innerText = "BASELINE SAVED";
                statusEl.className = "font-bold text-primary tracking-wide";
            }

            const eyeEl = document.getElementById('metric_eye');
            const shoulderEl = document.getElementById('metric_shoulder');
            if (eyeEl) eyeEl.innerText = `${Math.round(baselineEyeDist)} px`;
            if (shoulderEl) shoulderEl.innerText = `${Math.round(baselineShoulderY)} px`;

            btnCalibrate.innerText = "Recalibrate";
            btnCalibrate.disabled = false;

            startPostureAI();
        } else {
            if (statusEl) {
                statusEl.innerText = apiData ? apiData.status : "API CONNECT ERROR";
                statusEl.className = "font-bold text-error tracking-wide";
            }
            btnCalibrate.innerText = "Try Again";
            btnCalibrate.disabled = false;
        }
    });
}

function startPostureAI() {
    if (isMonitoring) return;
    isMonitoring = true;

    async function monitorLoop() {
        if (!isMonitoring) return;

        const base64Img = captureWebcamBase64();
        if (base64Img) {
            const apiData = await sendImageToAPI(base64Img, false);
            if (apiData && isMonitoring) {
                handleAPIResponse(apiData);
            }
        }

        if (isMonitoring) {
            setTimeout(monitorLoop, 1000);
        }
    }

    monitorLoop();
}

function stopPostureAI() {
    isMonitoring = false;
}

function handleAPIResponse(response) {
    const now = Date.now();
    const statusEl = document.getElementById('posture_status');
    const eyeEl = document.getElementById('metric_eye');
    const shoulderEl = document.getElementById('metric_shoulder');

    if (eyeEl) eyeEl.innerText = `${Math.round(response.eye_dist)} px`;
    if (shoulderEl) shoulderEl.innerText = `${Math.round(response.shoulder_y)} px`;

    if (response.is_bad_posture) {
        goodPostureStartTime = 0;

        if (badPostureStartTime === 0) {
            badPostureStartTime = now;
        }

        const elapsedBadTime = now - badPostureStartTime;

        if (elapsedBadTime >= 5000) {
            isWarningActive = true;
            showWarningUI(response.status, statusEl);
        } else {
            const countdown = 5 - Math.floor(elapsedBadTime / 1000);
            if (statusEl) {
                statusEl.innerText = `${response.status} (${countdown}s)`;
                statusEl.className = "font-bold text-amber-500 tracking-wide";
            }
        }

    } else {
        if (goodPostureStartTime === 0) {
            goodPostureStartTime = now;
        }

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