import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import re
import os  # 新增：用於處理路徑
from torchvision import transforms

# --- 1. 初始化 Flask App ---
app = Flask(__name__)

# --- 2. 加載模型 (與 train.py 相同的模型結構) ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 確保在 cpu 上運行
device = torch.device('cpu')
model = ConvNet().to(device)

# --- 修改路徑載入邏輯 ---
# 取得目前 app.py 檔案所在的資料夾絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'mnist_cnn.pth')

# 載入權重檔案
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. 圖像預處理 ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- 4. 路由設定 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 從 POST 請求中獲取 Base64 圖像數據
    img_data_url = request.values['img']
    # 解析 Base64
    img_data = re.sub('^data:image/.+;base64,', '', img_data_url)
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))

    # 預處理圖像
    image = transform(img).unsqueeze(0)

    # 進行預測
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        result = predicted.item()

    # 在 app.py 的 predict 函式中加入
        image = transform(img).unsqueeze(0)
        # 加入這行來觀察存出的圖片是否清晰且正確縮放
        # transforms.ToPILImage()(image.squeeze(0)).save('debug_input.png')

    return str(result)

# --- 5. 啟動伺服器 ---
if __name__ == '__main__':
    # 注意：在生產環境中，應使用 Gunicorn 或 uWSGI
    app.run(debug=True, port=5001)