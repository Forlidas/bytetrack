# ByteTrack

## 维护说明

### 项目定位
本仓库当前基于上游 `ifzhang/ByteTrack` 代码库，作为一个长期维护的私有工作区使用，主要用于：

- 在本地复现原始 ByteTrack 流程
- 在 Ubuntu + NVIDIA GPU 环境下维护一套稳定可运行的环境
- 承载后续算法修改与个人创新
- 保留足够文档，使本地文件丢失后仍能快速重建项目

### 当前已验证状态
下面这套环境已经在本机验证可运行：

- 主机系统：Ubuntu
- GPU：NVIDIA GeForce RTX 4080
- Conda 环境：`bytetrack-cu118`
- Python：`3.10`
- PyTorch：`2.0.1`
- torchvision：`0.15.2`
- 环境内 CUDA 运行时：`11.8`

一份新的官方 `bytetrack_x_mot17.pth.tar` 权重已经重新下载，并确认可以在 `videos/palace.mp4` 上产出有效的人员检测结果。

### 快速复现
#### 1. 激活环境
```bash
source ~/.bashrc
conda activate bytetrack-cu118
```

#### 2. 进入项目根目录
```bash
cd <ByteTrack_HOME>
```

#### 3. 确认官方 MOT17 demo 权重存在
期望路径：
```bash
pretrained/bytetrack_x_mot17.pth.tar
```

如果该文件不存在，请从上游模型链接重新下载官方权重。

#### 4. 运行已验证的 demo 命令
```bash
PYTHONPATH=$(pwd) python tools/demo_track.py video \
  --path ./videos/palace.mp4 \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --device gpu \
  --save_result
```

### 已验证的 Demo 输出
本地已经生成过一份可用输出，路径如下：

```bash
YOLOX_outputs/yolox_x_mix_det/track_vis/2026_07_12_22_45_22/palace.mp4
YOLOX_outputs/yolox_x_mix_det/track_vis/2026_07_12_22_45_22.txt
```

说明：

- 输出视频已经成功生成
- 跟踪结果 txt 非空
- 之前那份 379 MB 的 checkpoint 不适合本次 demo，不应继续直接复用

### 当前已经应用的本地兼容性修复
当前仓库已经包含最小化的 NumPy 兼容性修复，涉及：

- `yolox/tracker/byte_tracker.py`
- `yolox/tracker/matching.py`

原因：

- 老代码使用了 `np.float` 这类已废弃别名
- 新版 NumPy 在运行时会因此直接报错

### 这个私有仓库建议长期保留的内容
为了让项目在未来容易恢复，建议优先保留这些文件：

- 你修改过的源码
- 本 `README.md`
- 后续新增的 `environment.yml`
- 后续新增的一键脚本，例如 `scripts/setup.sh`、`scripts/run_demo.sh`
- 你自己的实验记录、配置文件、算法改动

不要只依赖这个仓库保存大型二进制文件。权重、数据集、运行输出等大文件建议单独备份。

### 当前忽略策略摘要
当前 `.gitignore` 默认忽略了这些重要运行产物：

- `pretrained/`
- `YOLOX_outputs/`
- `*.pth`
- `*.onnx`
- `*.engine`
- `*.npy`

这意味着模型权重和 demo 输出视频默认不会被提交。

### 后续维护建议
为了长期维护更稳妥，建议继续补充：

1. 增加 `environment.yml`
2. 增加一键安装与运行脚本
3. 记录权重下载链接与校验值
4. 用日期分节或 changelog 记录后续算法修改
5. 明确哪些生成产物继续留在 git 外部，哪些需要纳入版本管理

### 关于上游历史的说明
当前仓库仍然保留了上游 ByteTrack 的提交历史，这是正常的。

- 仓库所有者是你
- 历史提交作者仍然会显示为上游贡献者
- 你后续的修改会作为你自己的新提交叠加在上游历史之上

---

## 上游 README

#### ByteTrack 是一个简单、快速、效果很强的多目标跟踪器。

> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
>
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
>
> *[arXiv 2110.06864](https://arxiv.org/abs/2110.06864)*

## 演示链接
| Google Colab 演示 | Huggingface 演示 | YouTube 教程 | 原始论文 |
|:-----------------:|:----------------:|:------------:|:--------:|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bDilg4cmXFa8HCKHbsZ_p16p0vrhLyu0?usp=sharing)|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/bytetrack)|[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/QCG8QMhga9k)|[arXiv 2110.06864](https://arxiv.org/abs/2110.06864) |

- 已集成到 [Huggingface Spaces](https://huggingface.co/spaces)，前端使用 [Gradio](https://github.com/gradio-app/gradio)。

## 摘要
多目标跟踪（MOT）的目标是在视频中估计目标的边界框和身份。多数方法通过关联高于某个阈值的检测框来获得身份信息。而低分检测框，例如被遮挡的目标，往往会被直接丢弃，从而带来明显的漏检和轨迹碎片化问题。为了解决这个问题，我们提出了一种简单、有效且通用的关联方法：不只关联高分检测框，而是关联每一个检测框。对于低分检测框，我们利用其与轨迹片段之间的相似性来恢复真实目标，同时滤除背景误检。将该方法应用到 9 种不同的当前先进跟踪器上，都能稳定提升 IDF1，提升幅度在 1 到 10 个点之间。为了进一步推动 MOT 的 SOTA 表现，我们设计了一个简单而强大的跟踪器，命名为 ByteTrack。我们首次在 MOT17 测试集上实现了 80.3 MOTA、77.3 IDF1 和 63.1 HOTA，并在单张 V100 GPU 上达到了 30 FPS 的运行速度。

## 更新动态
- (2022.07) 论文被 ECCV 2022 接收。
- (2022.06) 百度 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mot/bytetrack) 给出了一份不错的复现实现。

## 跟踪性能
### MOT Challenge 测试集结果
| 数据集 | MOTA | IDF1 | HOTA | MT | ML | FP | FN | IDs | FPS |
|--------|------|------|------|------|------|------|------|------|------|
|MOT17|80.3|77.3|63.1|53.2%|14.5%|25491|83721|2196|29.6|
|MOT20|77.8|75.2|61.3|69.2%|9.5%|26249|87594|1223|13.7|

### MOT Challenge 测试集可视化结果
<img src="assets/MOT17-01-SDP.gif" width="400"/>   <img src="assets/MOT17-07-SDP.gif" width="400"/>
<img src="assets/MOT20-07.gif" width="400"/>   <img src="assets/MOT20-08.gif" width="400"/>

## 安装
### 1. 在主机上安装
步骤 1：安装 ByteTrack。
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

步骤 2：安装 [pycocotools](https://github.com/cocodataset/cocoapi)。

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

步骤 3：其他依赖。
```shell
pip3 install cython_bbox
```

### 2. Docker 构建
```shell
docker build -t bytetrack:latest .

# 启动示例
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
```

## 数据准备
下载 [MOT17](https://motchallenge.net/)、[MOT20](https://motchallenge.net/)、[CrowdHuman](https://www.crowdhuman.org/)、[Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)、[ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)，并按如下结构放到 `<ByteTrack_HOME>/datasets`：

```
datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Cityscapes
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
            └——————eth01
            └——————...
            └——————eth07
```

然后需要将这些数据集转换成 COCO 格式，并混合不同训练数据：

```shell
cd <ByteTrack_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
```

在混合不同数据集之前，需要按照 [mix_xxx.py](https://github.com/ifzhang/ByteTrack/blob/c116dfc746f9ebe07d419caa8acba9b3acfa79a6/tools/mix_data_ablation.py#L6) 中的操作创建数据目录和链接。最后可以执行：

```shell
cd <ByteTrack_HOME>
python3 tools/mix_data_ablation.py
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py
```

## 模型库

### 消融模型
在 CrowdHuman 和 MOT17 half train 上训练，在 MOT17 half val 上评估。

| 模型 | MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|ByteTrack_ablation [[google]](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing), [[baidu(code:eeo8)]](https://pan.baidu.com/s/1W5eRBnxc4x9V8gm7dgdEYg) | 76.6 | 79.3 | 159 | 29.6 |

### MOT17 测试模型
在 CrowdHuman、MOT17、Cityperson 和 ETHZ 上训练，在 MOT17 train 上评估。

- **标准模型**

| 模型 | MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot17 [[google]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [[baidu(code:ic0i)]](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw) | 90.0 | 83.3 | 422 | 29.6 |
|bytetrack_l_mot17 [[google]](https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing), [[baidu(code:1cml)]](https://pan.baidu.com/s/1242adimKM6TYdeLU2qnuRA) | 88.7 | 80.7 | 460 | 43.7 |
|bytetrack_m_mot17 [[google]](https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing), [[baidu(code:u3m4)]](https://pan.baidu.com/s/1fKemO1uZfvNSLzJfURO4TQ) | 87.0 | 80.1 | 477 | 54.1 |
|bytetrack_s_mot17 [[google]](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing), [[baidu(code:qflm)]](https://pan.baidu.com/s/1PiP1kQfgxAIrnGUbFP6Wfg) | 79.2 | 74.3 | 533 | 64.5 |

- **轻量模型**

| 模型 | MOTA | IDF1 | IDs | 参数量(M) | FLOPs(G) |
|------------|-------|------|------|------|-------|
|bytetrack_nano_mot17 [[google]](https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing), [[baidu(code:1ub8)]](https://pan.baidu.com/s/1dMxqBPP7lFNRZ3kFgDmWdw) | 69.0 | 66.3 | 531 | 0.90 | 3.99 |
|bytetrack_tiny_mot17 [[google]](https://drive.google.com/file/d/1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju/view?usp=sharing), [[baidu(code:cr8i)]](https://pan.baidu.com/s/1jgIqisPSDw98HJh8hqhM5w) | 77.1 | 71.5 | 519 | 5.03 | 24.45 |

### MOT20 测试模型
在 CrowdHuman 和 MOT20 上训练，在 MOT20 train 上评估。

| 模型 | MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot20 [[google]](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing), [[baidu(code:3apd)]](https://pan.baidu.com/s/1bowJJj0bAnbhEQ3_6_Am0A) | 93.4 | 89.3 | 1057 | 17.5 |

## 训练
YOLOX 的 COCO 预训练模型可以从它们的 [model zoo](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) 下载。下载后，将其放到 `<ByteTrack_HOME>/pretrained`。

- **训练消融模型（MOT17 half train + CrowdHuman）**

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

- **训练 MOT17 测试模型（MOT17 train + CrowdHuman + Cityperson + ETHZ）**

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

- **训练 MOT20 测试模型（MOT20 train + CrowdHuman）**

对于 MOT20，需要将边界框裁剪到图像内部。

在 [data_augment.py 第 134-135 行](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment.py#L134)、[mosaicdetection.py 第 122-125 行](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122)、[mosaicdetection.py 第 217-225 行](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217)、[boxes.py 第 115-118 行](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115) 增加 clip 操作。

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

- **训练自定义数据集**

首先，需要将你的数据集准备成 COCO 格式。可以参考 [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) 或 [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py)。然后，需要为你的数据集创建一个 Exp 文件，可参考 [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) 的训练 Exp 文件。不要忘记修改该 Exp 文件中的 `get_data_loader()` 和 `get_eval_loader()`。最后执行：

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

## 跟踪

- **在 MOT17 half val 上评估**

运行 ByteTrack：

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```

使用我们的预训练模型可以获得 76.6 MOTA。

运行其他跟踪器：
```shell
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
```

- **测试 MOT17**

运行 ByteTrack：

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/interpolation.py
```

将 txt 文件提交到 [MOTChallenge](https://motchallenge.net/) 网站，你可以得到 79+ MOTA（如果想达到 80+ MOTA，需要仔细调节测试图像尺寸和每个序列的高分检测阈值）。

- **测试 MOT20**

我们对 MOT20-04、MOT20-07 使用输入尺寸 1600 x 896，对 MOT20-06、MOT20-08 使用 1920 x 736。你可以在 [yolox_x_mix_mot20_ch.py](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_mix_mot20_ch.py) 中修改。

运行 ByteTrack：

```shell
cd <ByteTrack_HOME>
python3 tools/track.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/bytetrack_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --match_thresh 0.7 --mot20
python3 tools/interpolation.py
```

将 txt 文件提交到 [MOTChallenge](https://motchallenge.net/) 网站，你可以得到 77+ MOTA（如果想得到更高的 MOTA，需要仔细调节测试图像尺寸和每个序列的高分检测阈值）。

## 将 BYTE 应用于其他跟踪器
参见 [tutorials](https://github.com/ifzhang/ByteTrack/tree/main/tutorials)。

## 将 BYTE 与其他检测器结合
假设你已经从其他检测器拿到了检测结果 `dets`（格式为 `x1, y1, x2, y2, score`），你可以直接将检测结果传入 `BYTETracker`（你需要先根据自己的检测结果格式，修改 [byte_tracker.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) 里的部分后处理代码）：

```python
from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)
for image in images:
   dets = detector(image)
   online_targets = tracker.update(dets, info_imgs, img_size)
```

你可以从 `online_targets` 中获得每一帧的跟踪结果。可以参考 [mot_evaluators.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/evaluators/mot_evaluator.py) 中如何把检测结果传给 `BYTETracker`。

## 演示

<img src="assets/palace_demo.gif" width="600"/>

```shell
cd <ByteTrack_HOME>
python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
```

## 部署

1. [ONNX 导出与 ONNXRuntime](./deploy/ONNXRuntime)
2. [Python 版 TensorRT](./deploy/TensorRT/python)
3. [C++ 版 TensorRT](./deploy/TensorRT/cpp)
4. [C++ 版 ncnn](./deploy/ncnn/cpp)
5. [Deepstream](./deploy/DeepStream)

## 引用

```text
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## 致谢

本代码大量借鉴自 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、[FairMOT](https://github.com/ifzhang/FairMOT)、[TransTrack](https://github.com/PeizeSun/TransTrack) 和 [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp)，在此表示感谢。
