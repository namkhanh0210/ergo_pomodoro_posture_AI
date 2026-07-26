const video = document.getElementById('webcam-video');
const resultImage = document.getElementById('result-image');
const statusDisplay = document.getElementById('status-display');
const btnCalibrate = document.getElementById('btn-calibrate');

let isProcessing = false;
let isCalibration = false;
let baselineEyeDist = 0.0;
let baselineShoulderY = 0.0;

const API_URL = '/api/analyze_frame';

async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
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
    if (isProcessing || video.videoWidth === 0) return;
    
    isProcessing = true;
    
    try {
        const base64Data = captureCompressedBase64();
        
        const payload = {
            image_base64: base64Data,
            is_calibration: isCalibration,
            baseline_eye_dist: baselineEyeDist,
            baseline_shoulder_y: baselineShoulderY
        };

        const response = await fetch(API_URL, {
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
        statusDisplay.innerText = "Connection error";
    } finally {
        isProcessing = false;
    }
}

if (btnCalibrate) {
    btnCalibrate.addEventListener('click', () => {
        isCalibration = true;
        statusDisplay.innerText = "Calibrating...";
    });
}

initWebcam().then(() => {
    video.addEventListener('loadeddata', () => {
        setInterval(processFrame, 2000);
    });
});
