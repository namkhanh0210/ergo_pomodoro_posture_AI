const video = document.getElementById('webcam-video');
const resultImage = document.getElementById('result-image');
const statusDisplay = document.getElementById('status-display');
const btnCalibrate = document.getElementById('btn-calibrate');

const btnSkip = document.getElementById('btn-skip');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const userHeightInput = document.getElementById('user-height');
const stepAssessment = document.getElementById('step-assessment');
const stepMonitor = document.getElementById('step-monitor');

let isProcessing = false;
let isCalibration = false;
let baselineEyeDist = 0.0;
let baselineShoulderY = 0.0;

const API_ANALYZE = '/api/analyze_frame';
const API_ASSESS = '/api/assess_desk';

function goToMonitorStep() {
    if (stepAssessment) stepAssessment.style.display = 'none';
    if (stepMonitor) stepMonitor.style.display = 'block';
    initWebcam();
}

if (btnSkip) {
    btnSkip.addEventListener('click', (e) => {
        e.preventDefault();
        goToMonitorStep();
    });
}

if (btnUpload && fileInput) {
    btnUpload.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });

    fileInput.addEventListener('change', async () => {
        if (fileInput.files.length === 0) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('user_height', userHeightInput ? userHeightInput.value : 170);

        try {
            if (statusDisplay) statusDisplay.innerText = "Analyzing desk...";
            const response = await fetch(API_ASSESS, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            console.log("Assessment Result:", data);

            goToMonitorStep();
        } catch (err) {
            alert("Upload failed. Please try again.");
        }
    });
}

async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => {
            setInterval(processFrame, 2000);
        }, { once: true });
    } catch (err) {
        alert("Webcam access denied or unavailable.");
    }
}

function captureCompressedBase64() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const scale = 480 / Math.max(video.videoWidth, video.videoHeight);
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.45);
}

async function processFrame() {
    if (isProcessing || !video || video.videoWidth === 0) return;
    
    isProcessing = true;
    
    try {
        const base64Data = captureCompressedBase64();
        
        const payload = {
            image_base64: base64Data,
            is_calibration: isCalibration,
            baseline_eye_dist: baselineEyeDist,
            baseline_shoulder_y: baselineShoulderY
        };

        const response = await fetch(API_ANALYZE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.image) {
            resultImage.src = data.image;
        }
        
        if (data.status) {
            statusDisplay.innerText = data.status;
        }

        if (isCalibration && data.is_success) {
            baselineEyeDist = data.eye_dist;
            baselineShoulderY = data.shoulder_y;
            isCalibration = false;
            statusDisplay.style.color = "green";
        } else if (data.is_bad_posture) {
            statusDisplay.style.color = "red";
        } else {
            statusDisplay.style.color = "black";
        }

    } catch (error) {
        if (statusDisplay) statusDisplay.innerText = "Connection error";
    } finally {
        isProcessing = false;
    }
}

if (btnCalibrate) {
    btnCalibrate.addEventListener('click', (e) => {
        e.preventDefault();
        isCalibration = true;
        statusDisplay.innerText = "Calibrating...";
    });
}
