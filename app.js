const captureCanvas = document.createElement('canvas');
captureCanvas.width = 640; 
captureCanvas.height = 480;
const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });

let isProcessingFrame = false; 

let badPostureStartTime = null; 
const BAD_POSTURE_THRESHOLD = 5000;

const deskFileInput = document.getElementById('desk-file-input');
const step1Section = document.getElementById('step1-assessment');
const step2Section = document.getElementById('step2-monitor');
const startMonitorBtn = document.getElementById('start-monitor-btn');
const webcamElement = document.getElementById('webcam');
const warningUI = document.getElementById('warning-overlay');

const timerDisplay = document.getElementById('timer-display');
const progressBar = document.getElementById('progress-bar');
const sessionCounter = document.getElementById('session-counter');

deskFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const formData = new FormData();
        formData.append('image', file);

        console.log("Đang phân tích góc làm việc...");

        const response = await fetch('/api/assess_desk', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log("Điểm số:", data.score);

        if (data.audio_base64) {
            const audio = new Audio(data.audio_base64);
            audio.play().catch(err => console.log("Trình duyệt chặn tự động phát audio:", err));
            
            audio.onended = () => {
                audio.src = "";
            };
        }

    } catch (error) {
        console.error("Lỗi khi đánh giá góc làm việc:", error);
    }
});

startMonitorBtn.addEventListener('click', async () => {
    step1Section.style.display = 'none';
    step2Section.style.display = 'block';

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream;
        
        webcamElement.onloadedmetadata = () => {
            setInterval(monitorLoop, 2500); 
            startPomodoro();
        };
    } catch (error) {
        console.error("Không thể truy cập webcam:", error);
        alert("Vui lòng cấp quyền sử dụng camera để theo dõi tư thế!");
    }
});

function captureWebcamBase64() {
    if (!webcamElement || !webcamElement.srcObject || webcamElement.readyState !== 4) {
        return null;
    }
    
    captureCtx.drawImage(webcamElement, 0, 0, captureCanvas.width, captureCanvas.height);
    return captureCanvas.toDataURL('image/jpeg', 0.6);
}

async function monitorLoop() {
    if (isProcessingFrame) {
        return; 
    }

    const base64Image = captureWebcamBase64();
    if (!base64Image) return;

    isProcessingFrame = true;

    try {
        const response = await fetch('/api/analyze_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        if (data.is_bad_posture) {
            if (!badPostureStartTime) {
                badPostureStartTime = Date.now();
            } else {
                const timeBad = Date.now() - badPostureStartTime;
                if (timeBad > BAD_POSTURE_THRESHOLD) {
                    showWarning();
                }
            }
        } else {
            badPostureStartTime = null;
            hideWarning();
        }

    } catch (error) {
        console.error("Lỗi khi gửi frame:", error);
    } finally {
        isProcessingFrame = false;
    }
}

function showWarning() {
    warningUI.classList.add('active');
}

function hideWarning() {
    warningUI.classList.remove('active');
}

let pomodoroTimeLeft = 25 * 60;
let completedSessions = 0;

function startPomodoro() {
    const totalTime = pomodoroTimeLeft;

    const timerInterval = setInterval(() => {
        pomodoroTimeLeft--;

        const minutes = Math.floor(pomodoroTimeLeft / 60);
        const seconds = pomodoroTimeLeft % 60;
        timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

        const progressPercent = ((totalTime - pomodoroTimeLeft) / totalTime) * 100;
        progressBar.style.width = `${progressPercent}%`;

        if (pomodoroTimeLeft <= 0) {
            clearInterval(timerInterval);
            completedSessions++;
            sessionCounter.textContent = `Sessions: ${completedSessions}`;
            alert("Chúc mừng! Bạn đã hoàn thành 1 phiên Focus. Nghỉ ngơi 5 phút nhé!");
        }
    }, 1000);
}
