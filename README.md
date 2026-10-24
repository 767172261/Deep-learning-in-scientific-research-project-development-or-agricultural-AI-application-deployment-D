深度学习在科研、项目开发或农业AI应用部署 智慧农业、烟叶健康监测、自动巡检系统的开发 YOLOV8模型训练烟叶病虫害检测数据集
@[toc]
### **烟叶病虫害检测数据集（YOLO格式）**

| **项目** | **内容** |
|----------|----------|
| **数据集名称** | 烟叶病虫害检测数据集 |
| **图像总数** | 1,964 张 |
| **数据来源** | 大疆无人机拍摄（原始图像） |
| **预处理方式** | 图像裁剪、旋转、数据增强（如翻转、亮度调整等） |
| **图像质量** | 高清航拍图像，适用于小目标与病害识别 |
| **标注方式** | 使用专业标注软件进行人工标注 |
| **标注格式** | YOLO `.txt` 格式（每图对应一个标签文件） |
| **文件结构** | 包含 `images/`（图像）、`labels/`（标签）、`mydata.yaml`（配置文件） |
| **使用方式** | 可直接放入 YOLO 项目目录中训练，无需转换 |
| **适用模型** | YOLOv5、YOLOv8 等 YOLO 系列目标检测模型 |

---

### **类别定义表**

| 类别编号 | 类别名称（英文） | 类别名称（中文） | 说明 |
|----------|------------------|------------------|------|
| 0 | healthy | 健康烟叶 | 无病害症状的正常烟叶 |
| 1 | diseased | 病害烟叶 | 患有病虫害的烟叶（如花叶病、赤星病、虫蛀等） |

---

### **数据集特点**

- ✅ **真实场景**：基于无人机航拍，贴近实际农田监测环境  
- ✅ **专业标注**：由人工使用专业软件标注，精度高  
- ✅ **开箱即用**：包含完整 `images`、`labels` 和 `mydata.yaml`，可直接训练  
- ✅ **数据增强**：经过裁剪、旋转等增强处理，提升模型泛化能力  
- ✅ **农业实用性强**：适用于智慧农业、烟叶健康监测、自动巡检系统开发  

---

> 📌 **备注**：该数据集制作耗时较长，适合用于科研、项目开发或农业AI应用部署。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5dfe126e68db4a928eaa83b567cc2c66.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/623f04e63f4044c89f82fc404f4f999a.png)
 YOLOv8 的烟叶病虫害检测任务。**系统性地整理从环境搭建 → 模型训练 → 推理 → 评估**的全流程，代码示例，仅供参考。

---

## ✅ 一、项目结构建议


```
yolo_tobacco_project/
│
├── datasets/
│   └── tobacco/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── mydata.yaml
│
├── models/
│   └── yolov8n.pt          # 官方预训练权重
│
├── runs/                   # 训练输出目录（自动生成）
│
├── train.py                # 训练脚本
├── detect.py               # 推理脚本
├── val.py                  # 评估脚本
├── install_env.py          # 环境安装脚本
└── README.md
```

---

## ✅ 二、环境搭建全流程

### 1. 安装 CUDA 驱动（GPU 支持）

- 确保你的显卡支持 CUDA（NVIDIA）
- 前往 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads) 安装对应驱动
- 验证：打开终端，输入：
  ```bash
  nvidia-smi
  ```
  若显示 GPU 信息，则驱动安装成功。

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装后验证：
  ```bash
  conda --version
  ```

---

### 3. 创建 Python 虚拟环境并安装依赖

#### `install_env.py`（可运行脚本）

```python
# install_env.py
import subprocess
import sys

def install_packages():
    # 创建虚拟环境
    subprocess.run([
        "conda", "create", "-n", "tobacco_yolo", "python=3.9", "-y"
    ], shell=True)

    # 激活环境并安装包
    commands = [
        "conda activate tobacco_yolo",
        "pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
        "pip install ultralytics opencv-python PyQt5 matplotlib pillow"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    install_packages()
```

> 📌 **说明**：YOLOv8 推荐使用 `torch>=1.9`，这里使用 `torch 1.9.1 + CUDA 11.1` 兼容性好。

激活环境：
```bash
conda activate tobacco_yolo
```

---

## ✅ 三、数据集配置文件 `mydata.yaml`

你已有此文件，内容应如下：

```yaml
# mydata.yaml
train: ../datasets/tobacco/images/train
val: ../datasets/tobacco/images/val
test: ../datasets/tobacco/images/test

nc: 2
names: ['healthy', 'diseased']
```

> ✅ 确保路径正确，`nc` 是类别数，`names` 是类别名。

---

## ✅ 四、训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO

# 1. 加载预训练模型（官方 YOLOv8n）
model = YOLO('models/yolov8n.pt')  # 可替换为 yolov8s.pt, yolov8m.pt 等

# 2. 开始训练
results = model.train(
    data='datasets/tobacco/mydata.yaml',    # 数据集配置文件
    epochs=150,                             # 训练轮数
    imgsz=640,                              # 输入图像大小
    batch=16,                               # 批次大小（根据显存调整）
    name='tobacco_yolov8n_640',            # 实验名称
    device='cuda',                          # 使用 GPU
    workers=8,                              # 数据加载线程数
    optimizer='AdamW',                      # 优化器
    lr0=0.001,                              # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,                            # 早停轮数
    project='runs/tobacco',                # 保存路径
    amp=True,                               # 混合精度训练
    close_mosaic=30,                        # 关闭 Mosaic 增强的轮数
    cache='disk'                            # 缓存图像到磁盘，加快训练
)

# 3. 打印最终结果
print(results)
```

### 🚀 训练命令（终端运行）：
```bash
conda activate tobacco_yolo
python train.py
```

训练完成后，模型保存在：
```
runs/tobacco/tobacco_yolov8n_640/weights/best.pt
```

---

## ✅ 五、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2

# 1. 加载训练好的模型
model = YOLO('runs/tobacco/tobacco_yolov8n_640/weights/best.pt')

# 2. 设置置信度阈值
conf_threshold = 0.3

# --- 选项1：检测单张图片 ---
def detect_image():
    results = model.predict(
        source='datasets/tobacco/images/test/img001.jpg',
        conf=conf_threshold,
        save=True,          # 保存结果图
        project='runs/detect',
        name='test_image'
    )
    print(f"检测完成，结果保存在 runs/detect/test_image/")

# --- 选项2：检测整个文件夹 ---
def detect_folder():
    results = model.predict(
        source='datasets/tobacco/images/test/',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='test_batch'
    )
    print(f"批量检测完成，结果保存在 runs/detect/test_batch/")

# --- 选项3：调用摄像头实时检测 ---
def detect_camera():
    results = model.predict(
        source=0,  # 0 表示默认摄像头
        conf=conf_threshold,
        show=True,  # 实时显示
        stream=True
    )
    for r in results:
        pass  # 按 'q' 退出

# --- 选项4：检测视频文件 ---
def detect_video():
    results = model.predict(
        source='test_video.mp4',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='video_result'
    )

if __name__ == "__main__":
    detect_folder()  # 可切换为 detect_image(), detect_camera(), detect_video()
```

---

## ✅ 六、评估代码：`val.py`

评估训练好的模型在验证集上的性能（mAP, Precision, Recall 等）

```python
# val.py
from ultralytics import YOLO

# 1. 加载训练好的模型
model = YOLO('runs/tobacco/tobacco_yolov8n_640/weights/best.pt')

# 2. 在验证集上评估
metrics = model.val(
    data='datasets/tobacco/mydata.yaml',
    imgsz=640,
    batch=16,
    conf=0.001,      # 低置信度阈值用于评估
    iou=0.6,         # IoU 阈值
    device='cuda'
)

# 3. 打印评估结果
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
print("F1:", metrics.box.f1.mean())

# 结果图保存在 runs/val/tobacco_yolov8n_640/
```

---

## ✅ 七、可视化训练结果

训练完成后，`runs/tobacco/tobacco_yolov8n_640/` 目录会生成：

- `results.png`：训练损失、mAP 曲线
- `confusion_matrix.png`：混淆矩阵
- `PR_curve.png`：精确率-召回率曲线
- `F1_curve.png`：F1 分数曲线

这些图可帮助你分析模型性能。

---

## ✅ 八、常见问题与建议

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | 减小 `batch`（如 8 或 4），或使用 `yolov8n` 小模型 |
| 训练慢 | 使用 `cache='disk'` 或 `cache='ram'` 加快数据加载 |
| 过拟合 | 增加 `augment=True`，或减少 `epochs` |
| 检测效果差 | 检查标注质量，调整 `conf` 和 `iou` 阈值 |

---

## ✅ 九、一键运行建议

同学你可以创建一个 `run_all.py` 脚本：
```python
# run_all.py
import subprocess

subprocess.run(["python", "train.py"])
subprocess.run(["python", "val.py"])
subprocess.run(["python", "detect.py"])
```

---




