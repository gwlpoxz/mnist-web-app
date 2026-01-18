const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const resultSpan = document.getElementById('result');

let isDrawing = false;

// 1. 初始化畫布樣式
function initCanvas() {
    // 確保畫布背景是全黑 (MNIST 模型通常訓練於黑底白字)
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 設定畫筆為白色
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 15; // 稍微調細一點點，適合 280x280 轉 28x28
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

initCanvas();

// 2. 取得座標的輔助函式
function getCoords(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

// 3. 繪圖動作
function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getCoords(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    // 防止手機滑動頁面
    if (e.type === 'touchstart') e.preventDefault();
}

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
}

function draw(e) {
    if (!isDrawing) return;
    if (e.type === 'touchmove') e.preventDefault();

    const { x, y } = getCoords(e);
    ctx.lineTo(x, y);
    ctx.stroke();
}

// 4. 清除畫布
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultSpan.textContent = '-';
}

// 5. 辨識功能
async function predict() {
    const imageDataUrl = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `img=${encodeURIComponent(imageDataUrl)}`
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.text();
        resultSpan.textContent = result;

    } catch (error) {
        console.error('Error during prediction:', error);
        resultSpan.textContent = '錯誤';
    }
}

// --- 事件監聽 ---
// 滑鼠事件
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseout', stopDrawing);

// 觸控事件 (支援手機)
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchend', stopDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });

predictBtn.addEventListener('click', predict);
clearBtn.addEventListener('click', clearCanvas);
