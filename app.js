// ==========================================
// 1. KHỞI TẠO BIẾN & PHẦN TỬ DOM
// ==========================================
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

const BACKEND_URL = "https://ergopomodoropostureai.namkhanhnguyenquang.workers.dev";
const MODEL_PATH = "./yolov8n-pose.onnx";

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

// Canvas ẩn dùng cho việc xử lý YOLO Frame
const yoloCanvas = document.createElement('canvas');
yoloCanvas.width = 640;
yoloCanvas.height = 640;
const yoloCtx = yoloCanvas.getContext('2d', { willReadFrequently: true });

// ==========================================
// 2. HÀM XỬ LÝ ẢNH & CHUYỂN BƯỚC UI
// ==========================================
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
        await initYoloONNX();
        startPostureAI();
    }
};

// ==========================================
// 3. EVENT LISTENERS SETUP BÀN LÀM VIỆC
// ==========================================
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

            if (data.audio_base64) {
                const audio = new Audio(data.audio_base64);
                audio.play().catch(err => console.log("Audio autoplay blocked:", err));
                audio.onended = () => { audio.src = ""; };
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

// ==========================================
// 4. CHỨC NĂNG WEBCAM & ONNX YOLO POSE
// ==========================================
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

async function initYoloONNX() {
    if (sessionONNX) return true;
    const statusEl = document.getElementById('posture_status');
    if (statusEl) {
        statusEl.innerText = "LOADING YOLO MODEL...";
        statusEl.className = "font-bold text-amber-500 tracking-wide animate-pulse";
    }

    try {
        sessionONNX = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm']
        });
        console.log("✅ [ONNX Loaded] Inputs:", sessionONNX.inputNames, "Outputs:", sessionONNX.outputNames);
        if (statusEl) {
            statusEl.innerText = "YOLO READY (CALIBRATE REQUIRED)";
            statusEl.className = "font-bold text-primary tracking-wide";
        }
        return true;
    } catch (err) {
        console.error("❌ [ONNX Load Error]:", err);
        if (statusEl) {
            statusEl.innerText = "MODEL LOAD ERROR";
            statusEl.className = "font-bold text-error tracking-wide";
        }
        return false;
    }
}

function preprocessWebcamToTensor() {
    if (!webcam || webcam.readyState < 2 || webcam.videoWidth === 0) return null;

    yoloCtx.drawImage(webcam, 0, 0, 640, 640);
    const imageData = yoloCtx.getImageData(0, 0, 640, 640);
    const data = imageData.data;

    const float32Data = new Float32Array(1 * 3 * 640 * 640);

    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i] = data[i * 4] / 255.0;               // R
        float32Data[640 * 640 + i] = data[i * 4 + 1] / 255.0; // G
        float32Data[2 * 640 * 640 + i] = data[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
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
                    // Dùng linh hoạt tên input layer thay vì gán cứng 'images'
                    const inputName = sessionONNX.inputNames[0];
                    const feeds = {};
                    feeds[inputName] = inputTensor;

                    const results = await sessionONNX.run(feeds);
                    const outputName = sessionONNX.outputNames[0];
                    const outputTensor = results[outputName];

                    if (outputTensor && outputTensor.data) {
                        processYoloOutput(outputTensor.data);
                    }
                } catch (e) {
                    console.error("⚠️ [ONNX Execution Error]:", e);
                }
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

function processYoloOutput(data) {
    let maxScore = -1;
    let bestIdx = -1;

    // 1. Quét tìm Bounding Box có độ tin cậy cao nhất (Hạ ngưỡng xuống 0.10)
    for (let i = 0; i < 8400; i++) {
        const score = data[4 * 8400 + i];
        if (score > maxScore && score > 0.10) {
            maxScore = score;
            bestIdx = i;
        }
    }

    if (bestIdx === -1) return; // Không thấy người vượt ngưỡng score 0.10

    const getKP = (kpIdx) => {
        const offset = 5 + kpIdx * 3;
        return {
            x: data[offset * 8400 + bestIdx],
            y: data[(offset + 1) * 8400 + bestIdx],
            conf: data[(offset + 2) * 8400 + bestIdx]
        };
    };

    const nose = getKP(0);
    const leftEye = getKP(1);
    const rightEye = getKP(2);
    const leftShoulder = getKP(5);
    const rightShoulder = getKP(6);

    const minConf = 0.08; // Hạ ngưỡng khớp cơ thể xuống 0.08

    // 2. Tính toán khoảng cách giữa 2 mắt
    let currEyeDist = 0;
    if (leftEye.conf >= minConf && rightEye.conf >= minConf) {
        currEyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
    } else if (nose.conf >= minConf && leftEye.conf >= minConf) {
        currEyeDist = Math.hypot(leftEye.x - nose.x, leftEye.y - nose.y) * 2;
    } else if (nose.conf >= minConf && rightEye.conf >= minConf) {
        currEyeDist = Math.hypot(rightEye.x - nose.x, rightEye.y - nose.y) * 2;
    }

    // 3. Tính toán vị trí và chiều rộng Vai
    let currShoulderY = 0;
    let currShoulderWidth = 0;

    if (leftShoulder.conf >= minConf && rightShoulder.conf >= minConf) {
        currShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        currShoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
    } else if (leftShoulder.conf >= minConf) {
        currShoulderY = leftShoulder.y;
    } else if (rightShoulder.conf >= minConf) {
        currShoulderY = rightShoulder.y;
    } else if (nose.conf >= minConf) {
        currShoulderY = nose.y + 120;
    }

    // 4. CƠ CHẾ FALLBACK: Ước tính nếu 1 trong 2 thông số bị thiếu
    if (currEyeDist === 0 && currShoulderWidth > 0) {
        currEyeDist = currShoulderWidth / 2.5;
    } else if (currShoulderWidth === 0 && currEyeDist > 0) {
        currShoulderWidth = currEyeDist * 2.5;
    }

    // Nếu cả 2 đều bằng 0 mới bỏ qua frame này
    if (currEyeDist === 0 && currShoulderWidth === 0) return;

    // 5. Làm mượt chỉ số bằng Moving Average
    if (smoothedEyeDist === 0) {
        smoothedEyeDist = currEyeDist;
        smoothedShoulderY = currShoulderY;
        smoothedShoulderWidth = currShoulderWidth;
    } else {
        smoothedEyeDist = smoothedEyeDist * 0.75 + currEyeDist * 0.25;
        smoothedShoulderY = smoothedShoulderY * 0.75 + currShoulderY * 0.25;
        smoothedShoulderWidth = smoothedShoulderWidth * 0.75 + currShoulderWidth * 0.25;
    }

    // Cập nhật lên UI
    const eyeEl = document.getElementById('metric_eye');
    const shoulderEl = document.getElementById('metric_shoulder');
    if (eyeEl) eyeEl.innerText = `${Math.round(smoothedEyeDist)} px`;
    if (shoulderEl) shoulderEl.innerText = `${Math.round(smoothedShoulderY)} px`;

    if (!isCalibrated) return;

    // 6. Kiểm tra tư thế sau khi đã Calibrate
    const currentRatio = smoothedEyeDist / (smoothedShoulderWidth || 1);
    const baselineRatio = baselineEyeDist / (baselineShoulderWidth || 1);

    let isBadPosture = false;
    let statusText = "GOOD POSTURE";

    if (currentRatio > baselineRatio * 1.2) {
        isBadPosture = true;
        statusText = "TOO CLOSE TO SCREEN";
    } else if (smoothedShoulderY > baselineShoulderY + 20) {
        isBadPosture = true;
        statusText = "SLOUCHING DETECTED";
    }

    handlePostureStatus(isBadPosture, statusText);
}

// ==========================================
// 5. XỬ LÝ CALIBRATE TƯ THẾ
// ==========================================
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

        // Chờ 2.5 giây để webcam & AI quét lấy dữ liệu chuẩn
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

// ==========================================
// 6. CẢNH BÁO TƯ THẾ XẤU (POSTURE WARNING)
// ==========================================
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

// ==========================================
// 7. POMODORO TIMER LOGIC
// ==========================================
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

// Khởi chạy timer ban đầu
updateTimerDisplay();
