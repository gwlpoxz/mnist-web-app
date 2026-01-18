import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. 參數設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# --- 2. 數據準備 (Data Preparation) ---
# 將圖片轉為 Tensor 並進行標準化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. 神經網路搭建 (Model Architecture) ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 第一層卷積：輸入 1 channel (灰階), 輸出 32, 核心 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # 第二層卷積：輸入 32, 輸出 64, 核心 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        # 全連接層
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

model = ConvNet().to(device)

# --- 4. 訓練與損失函數 (Training) ---
# 使用 PyTorch 2.0+ 的編譯加速 (選配)
# model = torch.compile(model) 

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

# --- 5. 結果導出與保存 ---
def save_model():
    # 保存模型權重
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("\n模型已儲存至 mnist_cnn.pth")

if __name__ == "__main__":
    train()
    save_model()