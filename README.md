# D435i + QR 空间两点欧氏距离测量系统

本项目使用 Intel RealSense D435i 的 Left IR 与 Depth 流，实时计算两个二维码物理几何中心在相机坐标系中的三维位置，并输出两点之间的空间欧氏距离。默认产品算法是“QR 内部区域点云 + RANSAC 平面 + SVD 重拟合 + 中心射线与平面求交”，不会把二维码中心单个 Depth Pixel 当作最终测量结果。

> 当前状态：核心数学、记录与离线评估代码已有自动化测试；本仓库当前尚未在真实 D435i、具体固件、USB 主控和真实场景上完成端到端验证，也没有得到 Ground Truth 精度结论。请完成本文的真机与 1000 帧验收后再把结果用于工程决策。

## 1. 测量原理

处理链路如下：

```text
D435i synchronized frameset
        │
        ├── native Z16 Depth ───────────────────────────────┐
        │                                                   │
        └── Left IR1 ──安全映射到 Depth viewport──> QR 检测 │
                                                        │   │
                                  QR 四角与对角线交点 ───┘   │
                                                        │   │
                                       收缩 QR polygon ROI  │
                                                        │   │
                                       有效 Depth pixels ───┘
                                                │
                                      SDK 反投影为 Nx3 点云
                                                │
                              RANSAC 内点 + 全部内点 SVD 重拟合
                                                │
                                     QR 中心 camera ray × plane
                                                │
                                      QR_A XYZ 与 QR_B XYZ
                                                │
                                D = ||P_B - P_A||₂ + 时序统计
```

二维码图像中心严格定义为投影四边形两条对角线的交点，而不是四角坐标的算术平均：

```text
Line(q0, q2) ∩ Line(q1, q3)
```

QR 平面写为：

```text
nᵀP + d = 0,  ||n|| = 1
```

将二维码中心像素按深度 1 m 反投影得到射线方向 `R`，射线为 `P(t)=tR`，交点为：

```text
t = -d / (nᵀR)
P_QR = tR
```

这种方法利用二维码内部多个深度点，能排除部分飞点、孔洞、立体匹配错误和背景点。RANSAC 只负责选择内点，最终平面参数由全部内点做 SVD 重拟合。

## 2. 坐标系与单位

三维结果位于 D435i 原生 Depth 光学坐标系：

- 原点：Depth 成像器的光学中心；
- `+X`：从相机视角向右；
- `+Y`：从相机视角向下；
- `+Z`：从相机向前；
- `XYZ`、平面残差和算法内部距离：米；
- UI、CSV/JSONL 中的距离和 Plane RMS：毫米；
- 倾角：度；二维码图像尺寸：像素。

坐标约定可参考 RealSense 官方的 [Projection in RealSense SDK 2.0](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)。本项目不包含相机坐标到机器人、世界或机械治具坐标的外参变换。

## 3. 软硬件环境

### 硬件

- Intel RealSense D435i；
- 稳定的 USB 3.x 数据链路；
- 两个 payload 分别为 `QR_A`、`QR_B` 的平面二维码；
- 第一轮验证建议二维码边长 80～120 mm、相机距离 0.5～0.8 m、倾角小于 30°；
- 做 Accuracy 测试时，需要不确定度最好小于 0.5 mm 的 Ground Truth 工具或治具。

### 软件

- Python 3.10～3.12；项目默认固定为 Python 3.11，以减少 `pyrealsense2`、OpenCV 与平台 wheel 不匹配的概率；
- [uv](https://docs.astral.sh/uv/) 包管理器；项目固定使用 Python 3.11；
- Intel RealSense SDK 2.0 / librealsense；
- `pyrealsense2`、OpenCV、NumPy、PyYAML；
- pytest 仅用于测试，所有测试也采用 `unittest` 风格。

RealSense 安装入口：

- [librealsense 官方仓库](https://github.com/realsenseai/librealsense)
- [官方 Python Wrapper 安装说明](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md)
- [Ubuntu 官方安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md)

Linux 上还需要正确的 udev 权限；Windows 需要 SDK、Python 与架构匹配。虚拟机或 USB 转发会增加带宽和设备枚举问题，不建议用于首次真机验收。macOS 上 `pyrealsense2` wheel 和底层 USB 支持可能受 Python、CPU 架构与 SDK 版本限制，应先单独验证 SDK 能否识别相机。

## 4. 安装

本项目以 `pyproject.toml` 为唯一依赖声明，以 `uv.lock` 固定解析结果，并通过 `.python-version` 选择 Python 3.11。不再手工维护 `requirements.txt`。

先安装 uv：

- macOS：`brew install uv`
- Windows：`winget install --id=astral-sh.uv -e`
- 其他安装方式见 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)

在项目根目录安装 Python 并同步离线开发环境：

```bash
uv python install 3.11
uv sync --locked
```

`uv sync` 会自动创建项目专用的 `.venv`，并默认安装 `dev` 组中的 pytest，无需手动激活环境。

真机、bag 录制或回放需要 RealSense extra：

```bash
uv sync --locked --extra realsense
```

部署机器不需要测试依赖时：

```bash
uv sync --locked --no-dev --extra realsense
```

`pyrealsense2` 在 PyPI 上只有特定 Python、操作系统和 CPU 架构的 wheel。若 `--extra realsense` 报告没有兼容分发包，需要按 librealsense 官方说明构建 Python binding 或使用目标平台的本地 wheel；不要把依赖静默跳过后继续真机测量。

检查 Python 绑定：

```bash
uv run --locked python -c "import cv2, numpy, yaml; print('OpenCV', cv2.__version__)"
uv run --locked --extra realsense python -c "import pyrealsense2 as rs; print('pyrealsense2 OK')"
```

依赖变更使用 `uv add PACKAGE`，开发依赖使用 `uv add --dev PACKAGE`；检查锁文件是否同步可运行 `uv lock --check`。只有需要兼容旧部署工具时才从锁文件导出 requirements，例如 `uv export --locked --no-dev --extra realsense --no-emit-project -o requirements.txt`，导出文件不应手工编辑或提交为第二份依赖源。

真机运行前，建议先用 `realsense-viewer` 确认设备、固件、USB 链路、Depth 与 Left IR1 均可正常输出。能导入 `pyrealsense2` 不代表操作系统已经正确识别相机。

### Windows PowerShell 快速启动

Windows 真机环境建议先从 [librealsense Releases](https://github.com/realsenseai/librealsense/releases) 安装已发布的 SDK/Viewer，并用 `realsense-viewer` 确认 D435i、Depth 和 Left IR1 正常。SDK/Viewer 与 Python 的 `pyrealsense2` wheel 是两层独立检查，两者都要通过。

以下命令在 PowerShell 中执行；如果尚未安装 Git，请先安装 Git for Windows：

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
git clone https://github.com/snake-fan/D435i_Estimation_of_spatial_distance.git
Set-Location D435i_Estimation_of_spatial_distance
uv python install 3.11
uv sync --locked --extra realsense
uv run --locked --extra realsense python -c "import pyrealsense2 as rs; print(len(rs.context().devices), 'device(s)')"
uv run --locked --extra realsense python main.py --debug
```

最后一条命令会启动实时窗口。设备检查若显示 `0 device(s)`，应先处理 SDK、USB 3.x、驱动或设备枚举问题，而不是调整测量算法。部署环境若不需要 pytest，应在运行命令中同时保留 `--no-dev`，例如 `uv run --locked --no-dev --extra realsense python main.py --no-display`，防止 uv 重新加入开发依赖。

## 5. 生成 QR_A 与 QR_B

二维码 payload 区分大小写，必须精确为 `QR_A` 和 `QR_B`；系统不会按左右位置猜测 ID。

仓库提供固定参数的生成脚本；`qr` 是可选工具依赖，不会进入默认测量运行环境：

```bash
uv run --locked --extra qr python tools/generate_qr.py
```

图片默认生成到 `outputs/qr/QR_A.png` 与 `outputs/qr/QR_B.png`。脚本固定使用 QR Version 1、M 级纠错、4-module quiet zone，并拒绝静默覆盖已有文件；需要重建时加 `--overwrite`。

打印时建议：

- 保留二维码四周 quiet zone，不要裁掉白边；
- 本文的 80～120 mm 指**黑色有效码区**的边长，不含四周 4-module quiet zone；Version 1 的总图边长约为有效码区的 `29/21`，即约 110～166 mm；
- 以 100%/实际尺寸打印，关闭“适合页面”“缩放到纸张”等自动缩放；
- 使用平整、哑光、不透明的纸面或板材，避免反光膜和翘曲；
- 不要在 QR 内部打孔、贴胶带或覆盖文字；
- 两个二维码可以不同深度、不同姿态、不共面且不平行，但第一轮先控制倾角小于 30°；
- 打印后用尺测量黑色有效码区，并用手机分别扫码确认 payload 精确为 `QR_A`、`QR_B`；实时检测的最短边默认至少需要 35 px。

## 6. 配置

默认配置位于 [`config.yaml`](config.yaml)。所有算法距离阈值均使用米。

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `camera.width / height / fps` | `848 / 480 / 30` | Depth 与 Left IR1 请求规格 |
| `camera.emitter_enabled` | `false` | 是否启用红外投射器；散斑可能干扰 QR，默认关闭 |
| `camera.warmup_frames` | `30` | 真机启动、bag 首轮及 `--repeat-bag` 每轮开头丢弃的预热帧数 |
| `camera.serial_number` | `null` | 多相机时指定序列号 |
| `camera.frame_timeout_ms` | `5000` | 等待 frameset 超时 |
| `qr.expected_ids` | `[QR_A, QR_B]` | 必须是两个不同且非空的 payload |
| `qr.polygon_shrink_ratio` | `0.75` | QR polygon 向中心收缩比例 |
| `qr.min_edge_pixels` | `35` | QR 最短边像素门限 |
| `qr.corner_refinement` | `true` | 尝试 `cornerSubPix`；失败会回退原角点 |
| `plane.ransac_iterations` | `200` | RANSAC 迭代次数 |
| `plane.distance_threshold` | `0.004` | RANSAC 内点距离阈值，4 mm |
| `plane.min_points` | `80` | ROI 最少有效三维点数 |
| `plane.min_inlier_ratio` | `0.65` | 最低平面内点比例 |
| `plane.max_rms` | `0.003` | Plane 内点 RMS 门限，必须小于 RANSAC 内点阈值 |
| `plane.sample_stride` | `2` | ROI 像素二维采样步长 |
| `plane.random_seed` | `0` | RANSAC 随机种子；设为 `null` 可使用非确定性采样 |
| `plane.degenerate_epsilon` | `1e-12` | 退化平面判断容差 |
| `measurement.min_depth / max_depth` | `0.25 / 2.0` | 有效工作深度范围 |
| `measurement.min_valid_depth_ratio` | `0.35` | ROI 采样位置中成功反投影的最低比例 |
| `measurement.min_spatial_quadrants` | `3` | QR 中心四象限中至少要有几个象限得到深度支撑 |
| `measurement.min_points_per_quadrant` | `5` | 一个象限计为有支撑所需的最少采样点 |
| `measurement.warning_tilt_deg` | `45` | 达到此倾角后状态为 `WARNING` |
| `measurement.max_tilt_deg` | `60` | 超过此倾角后拒绝测量 |
| `measurement.ray_parallel_epsilon` | `1e-10` | 射线与平面近似平行判断容差 |
| `temporal.window_size` | `20` | 最近有效距离窗口 |
| `temporal.min_valid_frames` | `10` | `temporal.ready=true` 所需样本数 |
| `visualization.enabled` | `true` | 是否默认启用 OpenCV UI |
| `visualization.depth_min_m / depth_max_m` | `0.25 / 2.0` | Depth 热力图显示范围 |
| `recording.csv_enabled` | `false` | 为 `true` 时自动创建 UTC 时间戳 CSV；CLI `--record` 可显式指定新文件 |
| `recording.flush_every_row` | `true` | 每帧刷新 CSV，降低异常退出时的数据损失 |

配置加载会拒绝未知字段和非法范围，拼写错误不会被静默忽略。

## 7. 运行命令

以下命令均在项目根目录执行。每条真机与 bag 命令都显式使用 `--locked --extra realsense`，因此不会依赖之前某次环境同步是否选择了 RealSense extra，也不会在运行时改写锁文件。

### 真机实时测量

```bash
uv run --locked --extra realsense python main.py
```

指定配置：

```bash
uv run --locked --extra realsense python main.py --config config.yaml
```

显示更详细的检测和深度点信息：

```bash
uv run --locked --extra realsense python main.py --debug
```

### CSV 记录

```bash
uv run --locked --extra realsense python main.py --record outputs/measurements.csv
```

CSV 会记录有效与无效帧；无效数值留空，不会伪造为距离 0。`--record` 启动后立即写入，程序拒绝覆盖已有 CSV，请为每次实验使用新文件名。

### JSON Lines 输出

每帧写一个 JSON 对象：

```bash
uv run --locked --extra realsense python main.py --json-output outputs/results.jsonl
```

写到标准输出，适合管道处理：

```bash
uv run --locked --extra realsense python main.py --no-display --json-output -
```

JSONL 不是一个外层 JSON 数组；应逐行解析。

除 `--json-output -` 外，程序拒绝覆盖已有 JSONL。日志使用标准错误流，因此可以把标准输出中的 JSONL 单独接入下游进程。

### 录制 RealSense bag

```bash
mkdir -p data
uv run --locked --extra realsense python main.py --record-bag data/scene_01.bag
```

bag 记录当前启用的 Left IR1、Depth、标定、帧时间等 SDK 数据。为避免破坏实验数据，程序拒绝覆盖已存在的 `.bag`；父目录必须先存在。可以同时使用 `--record` 和 `--json-output` 记录算法结果。

`--record-bag` 与 `--bag` 互斥：前者只能录制真机，后者只做回放。

### 回放 bag

```bash
uv run --locked --extra realsense python main.py --bag data/scene_01.bag
```

回放与真机共用同一测量链。默认按有限、非循环数据源处理，正常 EOF 后退出；算法回归比较时建议把 `plane.random_seed` 固定为整数。bag 必须包含兼容的 Depth 与 Left IR1 流，默认还会核验其设备为 D435i。

本程序录制 bag 时，SDK 会把相机启动阶段也写入文件；回放首轮会按 `camera.warmup_frames` 跳过同样数量的帧，从而与真机首次进入算法的帧保持一致。使用 `--repeat-bag` 时，程序会根据设备时间戳或帧号回退识别循环重启，并在**每一轮**重新跳过该数量的帧，避免只在首轮预热而让后续循环的启动段进入测量。若回放的是已经人工裁掉启动段的外部 bag，请在专用配置中把 `camera.warmup_frames` 设为 `0`。

循环中必须至少存在一帧位于预热段之后；若一轮总帧数不大于 `camera.warmup_frames`，程序在检测到没有可输出的后预热帧时会抛出 `BagPlaybackError`，而不是无限跳帧。短 bag 回归应相应减小预热数或使用未裁剪、长度足够的录制文件。

默认回放会尽快处理每一帧，不按原始时间节奏丢帧。需要模拟录制时序时使用：

```bash
uv run --locked --extra realsense python main.py --bag data/scene_01.bag --bag-real-time
```

循环回放使用 `--repeat-bag`；无人值守时应同时指定 `--max-frames`，否则不会在 EOF 退出：

```bash
uv run --locked --extra realsense python main.py \
  --bag data/scene_01.bag \
  --repeat-bag \
  --no-display \
  --max-frames 3000
```

### 导出 QR ROI 点云

```bash
uv run --locked --extra realsense python main.py --dump-points
```

纯 flag 默认写入 `debug_points/`。指定目录：

```bash
uv run --locked --extra realsense python main.py --dump-points outputs/points
```

导出内容为每个 QR 最近一次可用 ROI 点云，字段是 `x,y,z,inlier`。同一 QR 的文件会被最新帧原子覆盖，坐标单位为米，`inlier` 为 `0/1`；该选项用于诊断平面拟合，不建议长期无人值守地频繁读取这些文件。

### Headless 与限定帧数

服务器、SSH 或无桌面环境必须关闭 OpenCV 窗口：

```bash
uv run --locked --extra realsense python main.py \
  --no-display \
  --record outputs/run_1000.csv \
  --json-output outputs/run_1000.jsonl \
  --max-frames 1000
```

`--max-frames N` 在处理 N 个输入帧后正常退出，适合自动化验收。Headless 模式没有键盘控制，因此应通过 CLI 提前配置输出路径。

也可对 bag 做固定帧回归：

```bash
uv run --locked --extra realsense python main.py \
  --bag data/scene_01.bag \
  --no-display \
  --max-frames 1000 \
  --record outputs/scene_01_replay.csv
```

如果 bag 少于指定帧数，会在正常 EOF 时提前结束。

### 设备选择与日志

多台真机时，CLI 可以临时覆盖配置中的序列号：

```bash
uv run --locked --extra realsense python main.py --serial 0123456789
```

`--serial` 只适用于真机，不能与 `--bag` 一起使用。默认严格核验 D435i 元数据；仅在你已经确认流、标定和误差行为兼容时，才允许其他 RealSense 型号：

```bash
uv run --locked --extra realsense python main.py --allow-compatible-device
```

这只关闭型号保护，不代表该设备已经通过本项目验收。日志级别可选 `DEBUG`、`INFO`、`WARNING`、`ERROR`：

```bash
uv run --locked --extra realsense python main.py --log-level DEBUG
uv run --locked python main.py --version
uv run --locked python main.py --help
```

正常完成返回码为 0；配置错误为 2，RealSense/相机源错误为 3，其他应用错误为 4，`Ctrl+C` 中断为 130，便于自动化脚本判断结果。

## 8. 实时界面与键盘

OpenCV UI 显示 IR 或 Depth 图、QR polygon、收缩 ROI、中心点、ID、XYZ、Plane RMS、内点率、倾角、像素尺寸、瞬时距离与时序统计。启用 `--debug` 后，每个 QR 还会显示 `Depth valid xx.x% Q=n/4`，分别对应 ROI 采样位置的有效反投影比例和当前结果的四象限支撑数。

| 按键 | 行为 |
| --- | --- |
| `Q` 或 `Esc` | 正常退出 |
| `R` | 暂停/恢复 CSV；若启动时未给 `--record`，首次按下会创建 UTC 时间戳命名的 `measurements_*.csv` |
| `D` | 在 Left IR 与 Depth 热力图之间切换 |

Depth 视图中超出显示范围或深度为 0 的像素以无效颜色显示，白线表示 QR ROI。`--debug` 只增加诊断细节，不改变产品测量方法。

## 9. 状态、质量门控与失败原因

只有两个二维码都有效时才输出距离并加入时序窗口：

- `GOOD`：所有硬门限通过，倾角低于 warning 门限；
- `WARNING`：结果仍有效，但至少一个 QR 倾角达到 warning 门限且未超过最大倾角；
- `INVALID`：任一硬门限失败，`distance_mm` 为 `null` 或 CSV 空值。

常见 `reject_reason`：

| 原因 | 含义与优先检查项 |
| --- | --- |
| `qr_not_found` | 没有解码到预期 payload；检查打印、照明、对焦、遮挡与 ID |
| `payload_mismatch` | 检测结果与当前期望 ID 不一致 |
| `invalid_qr_geometry` | 四角非有限、非凸或对角线退化 |
| `qr_too_small` | 最短边小于 `min_edge_pixels` |
| `not_enough_depth_points` | 收缩 ROI 中有效深度点不足 |
| `low_valid_depth_ratio` | 虽达到绝对点数，但有效深度只占 ROI 很小部分 |
| `poor_spatial_support` | 有效深度集中在中心一侧，不能安全外推到 QR 中心 |
| `deprojection_failed` | Depth scale、ROI 或 SDK 反投影调用无效 |
| `ransac_failed` | 没有得到非退化平面 |
| `low_inlier_ratio` | 平面候选存在，但内点比例太低 |
| `plane_rms_too_large` | 最终内点平面 RMS 超过门限 |
| `qr_tilt_too_large` | 倾角超过最大值或不可计算 |
| `invalid_intersection` | 中心射线与平面无有效前向交点 |
| `out_of_depth_range` | 求得中心点 Z 超出工作范围 |
| `depth_ir_shape_mismatch` | 当前帧的对齐 IR 与原生 Depth shape 不一致 |
| `processing_error` | 单 QR 处理发生未分类异常；用 `--debug` 查看 traceback |
| `invalid_point_xyz` | 成对距离阶段收到非有限或错误 shape 的三维点 |

成对结果会把 QR ID 加到原因前，例如 `QR_A:qr_not_found`。IR/Depth 对齐或 shape 失败属于帧源错误，日志中会显示 `FrameAlignmentError` 或 `FrameShapeMismatchError`，不能用伪造的距离继续运行。

`spatial_quadrants` 有两阶段语义：拟合前先用全部有效反投影点统计四象限，用于尽早拒绝深度只集中在 QR 一侧的 ROI；得到平面后，再仅用最终 RANSAC 内点重算并进行最终门控。成功结果以及带平面结果写入 UI、CSV/JSONL 的值是**拟合后的内点支撑数**；若在拟合前就被拒绝，则记录的是**拟合前的有效深度支撑数**。分析 `poor_spatial_support` 时应结合该 QR 是否已有 Plane RMS/内点率，不能把两个阶段的数值混为同一统计口径。

时序统计只接收有限、非负且有效的距离。默认最近 20 个有效值计算：

- Median：主稳定显示；
- Mean：均值；
- STD：总体标准差，`ddof=0`；
- MAD：`median(|Dᵢ - median(D)|)`。

至少 10 个有效样本后 `temporal.ready=true`。当前主程序使用时序模块默认 stale 策略：连续 10 个无效帧，或距离上次有效样本超过约 1000 ms 时清空旧窗口，避免 QR 消失后继续显示陈旧距离。

若主循环遇到 `FrameAcquisitionError`，该次采集没有可信设备时间戳，程序会立即清空时序窗口；采集恢复后的有效距离需要重新累计到 `temporal.min_valid_frames` 才会结束 warming。这样不会把超时前的旧窗口与恢复后的帧混合。

UI 在达到该门限前显示 `Temporal: warming`，stale 时显示 `Temporal: STALE`；不会把第一帧的中位数伪装成已经稳定的时序结果。

> STD 衡量重复性，不等于绝对误差。即使 `STD=0.7 mm`，系统仍可能稳定地偏差 6 mm；绝对精度必须通过 Ground Truth 的 Bias、MAE、RMSE 和 P95 Absolute Error 评价。

## 10. 输出 Schema

### CSV

CSV 每个输入帧一行，固定列如下：

| 列 | 单位/含义 |
| --- | --- |
| `wall_time_utc` | 写入时 UTC ISO 8601 时间 |
| `source_timestamp_ms` | RealSense 源时间戳，毫秒；不保证等于 UTC |
| `frame_number` | Depth 帧号 |
| `qr_a_id`, `qr_b_id` | 配置中的两个 payload |
| `qr_a_status`, `qr_b_status` | 每个 QR 独立的 `GOOD` / `WARNING` / `INVALID` |
| `qr_a_reject_reason`, `qr_b_reject_reason` | 每个 QR 独立的失败原因 |
| `qr_a_x_m`, `qr_a_y_m`, `qr_a_z_m` | QR_A 三维中心，米 |
| `qr_b_x_m`, `qr_b_y_m`, `qr_b_z_m` | QR_B 三维中心，米 |
| `distance_mm` | 瞬时三维欧氏距离，毫米 |
| `qr_a_plane_rms_mm`, `qr_b_plane_rms_mm` | 最终平面内点 RMS，毫米 |
| `qr_a_inlier_ratio`, `qr_b_inlier_ratio` | RANSAC 内点比例，0～1 |
| `qr_a_tilt_deg`, `qr_b_tilt_deg` | QR 平面倾角，度 |
| `qr_a_edge_pixels`, `qr_b_edge_pixels` | QR 最短图像边长，像素 |
| `qr_a_valid_depth_points`, `qr_b_valid_depth_points` | ROI 有效三维点数 |
| `qr_a_valid_depth_ratio`, `qr_b_valid_depth_ratio` | 有效反投影点数 / ROI 采样位置数 |
| `qr_a_spatial_quadrants`, `qr_b_spatial_quadrants` | 得到足够深度支撑的中心象限数，0～4 |
| `temporal_sample_count` | 当前有效窗口样本数 |
| `temporal_ready` | 是否达到最少有效样本数且未 stale |
| `temporal_median_mm` | 窗口中位数，毫米 |
| `temporal_mean_mm` | 窗口均值，毫米 |
| `temporal_std_mm` | 窗口总体标准差，毫米 |
| `temporal_mad_mm` | 窗口 MAD，毫米 |
| `method_a_distance_mm` | 研究 baseline A：中心单 Pixel Depth 距离，毫米 |
| `method_b_distance_mm` | 研究 baseline B：ROI 中位 Depth 距离，毫米 |
| `method_c_distance_mm` | 产品 Method C 距离，毫米；通常等于 `distance_mm` |
| `status` | `GOOD` / `WARNING` / `INVALID` |
| `reject_reason` | 分号分隔的明确失败原因 |

### JSONL

每行结构如下；无效或不可用数值使用 JSON `null`，不会输出 `NaN` 或 `Infinity`：

```json
{
  "source_timestamp_ms": 123456.789,
  "frame_number": 42,
  "qr_a": {
    "id": "QR_A",
    "valid": true,
    "status": "GOOD",
    "point_m": [0.123, -0.083, 0.756],
    "plane_rms_mm": 1.82,
    "inlier_ratio": 0.873,
    "tilt_deg": 23.1,
    "min_edge_pixels": 54.2,
    "valid_depth_points": 204,
    "valid_depth_ratio": 0.91,
    "spatial_quadrants": 4,
    "reject_reason": null
  },
  "qr_b": {
    "id": "QR_B",
    "valid": true,
    "status": "GOOD",
    "point_m": [-0.215, 0.104, 0.941],
    "plane_rms_mm": 2.31,
    "inlier_ratio": 0.846,
    "tilt_deg": 31.4,
    "min_edge_pixels": 48.6,
    "valid_depth_points": 181,
    "valid_depth_ratio": 0.87,
    "spatial_quadrants": 4,
    "reject_reason": null
  },
  "distance_mm": 427.36,
  "temporal": {
    "sample_count": 20,
    "ready": true,
    "median_mm": 427.11,
    "mean_mm": 427.19,
    "std_mm": 1.64,
    "mad_mm": 1.08
  },
  "status": "GOOD",
  "reject_reason": null
}
```

### Point cloud dump

`--dump-points` 为每个 QR 输出：

```text
x,y,z,inlier
0.1201,-0.0812,0.7510,1
0.1210,-0.0810,0.7521,0
```

`x/y/z` 为米；`inlier=1` 表示最终 RANSAC 内点集合。

## 11. 离线测试

纯数学、配置、记录与评估测试不需要 D435i，也不应在导入阶段要求 `pyrealsense2`。

运行全部 `unittest`：

```bash
uv run --locked python -m unittest discover -s tests -v
```

运行核心几何测试：

```bash
uv run --locked python -m unittest \
  tests.test_quadrilateral \
  tests.test_plane_svd \
  tests.test_plane_ransac \
  tests.test_ray_plane -v
```

使用 pytest：

```bash
uv run --locked python -m pytest
```

测试覆盖标准/透视/退化四边形、水平与倾斜平面、Gaussian noise + outlier 的 RANSAC、SVD 重拟合、法向 `z>=0` 约定、平行射线和欧氏距离相关基础行为。OpenCV 在当前平台不可导入时，依赖窗口的测试可以跳过，但真机运行前必须解决 OpenCV 安装问题。

## 12. Ground Truth 精度评估

先采集固定场景：

```bash
uv run --locked --extra realsense python main.py \
  --no-display \
  --max-frames 1000 \
  --record outputs/gt_500mm.csv
```

假设 QR 几何中心真实三维距离为 500.000 mm：

```bash
uv run --locked python -m evaluation.evaluate_ground_truth \
  outputs/gt_500mm.csv \
  --ground-truth-mm 500.000
```

默认只纳入 `GOOD` 和 `WARNING`。只评估 `GOOD`：

```bash
uv run --locked python -m evaluation.evaluate_ground_truth \
  outputs/gt_500mm.csv \
  --ground-truth-mm 500.000 \
  --include-status GOOD
```

也可通过 `--column` 选择其他毫米列。输出指标：样本数、测量 mean/median、重复性 STD、Bias、MAE、RMSE、P95 Absolute Error。

建议测试矩阵至少包含：

- 相机距离：0.3、0.5、0.8、1.0、1.2、1.5 m；
- QR 倾角：0°、30°、45°、60°；
- 图像位置：中心、中间区域、测量边界附近；
- 两点连线方向：主要沿 X、Y、Z 和 XYZ 混合，重点测试沿 Z。

设计阶段的误差量级只能作为实验规划参考，不能写成产品保证。最终精度应按场景分别报告 Bias、MAE、RMSE、P95，并同时报告 Ground Truth 自身不确定度。

## 13. Method A / B / C 比较

研究代码保留三个方法：

- Method A：中心单 Depth Pixel，只作 baseline；
- Method B：QR ROI 有效深度中位数，再反投影中心；
- Method C：RANSAC plane + SVD refit + ray-plane，本项目唯一默认产品路径。

`evaluation.baseline_methods` 提供 A/B 的研究函数。CSV 记录启用时，主程序会对同一输入帧同时生成：

```text
method_a_distance_mm
method_b_distance_mm
method_c_distance_mm
```

因此可以直接运行：

```bash
uv run --locked python -m evaluation.compare_methods \
  outputs/methods.csv \
  --ground-truth-mm 500.000 \
  --paired-only
```

对自定义列名使用可重复的 `--method NAME=COLUMN`：

```bash
uv run --locked python -m evaluation.compare_methods \
  outputs/custom_methods.csv \
  --ground-truth-mm 500.000 \
  --method A=single_pixel_mm \
  --method B=roi_median_mm \
  --method C=distance_mm
```

这些 baseline 列只用于离线对比，不会改变 UI、JSONL、质量门控或默认距离结果；产品输出始终使用 Method C。只评估 Method C 时，优先使用上一节的 `evaluate_ground_truth` 和 `distance_mm`。

默认情况下 `compare_methods` 会分别收集每列中的有限值，不同方法的 `sample_count` 可能不同；报告同时给出 `input_row_count` 与 `valid_rate`。`--paired-only` 只保留 A/B/C 三列都有限的共同帧，用于严格的同帧公平比较。

## 14. 真机 1000 帧验收

第一阶段目标先证明几何链正确、运行稳定、结果可诊断和重复性可接受；不要在没有 Ground Truth 的情况下宣称“绝对误差小于 5 mm”。

1. 固定 D435i，使用稳定 USB 3.x 端口，关闭自动移动的支架或柔性线缆影响。
2. 固定两个 80～120 mm QR，初始距离相机 0.5～0.8 m、倾角小于 30°。
3. 用 `realsense-viewer` 检查 Depth 和 Left IR1；记录相机型号、序列号、固件、USB 类型和环境条件。
4. 保留默认配置启动一次，核对日志中的流规格、depth scale、`fx/fy/ppx/ppy`、畸变参数、emitter 和 alignment strategy；先用 `--debug` 短时观察两个 QR 的 `Depth valid xx.x%` 与 `Q=n/4`，确认深度支撑不是只集中在局部。
5. 录制一份 bag，随后所有算法版本都回放同一份数据；固定 `plane.random_seed` 以减少随机差异。
6. 运行固定帧采集：

   ```bash
   mkdir -p data outputs
   uv run --locked --extra realsense python main.py \
     --no-display \
     --max-frames 1000 \
     --record outputs/validation_1000.csv \
     --record-bag data/validation_1000.bag
   ```

7. 确认程序正常退出、CSV 恰有 1000 个数据帧且没有用 0 填充无效距离。
8. 汇总并保存：有效帧率、状态/失败原因计数、Distance Mean/Median/STD/MAD/P95-P5、两 QR Plane RMS 分布、内点率、有效 Depth 点数与比例、`spatial_quadrants` 分布及 `poor_spatial_support` 次数、倾角和像素尺寸。
9. 切换 emitter OFF/ON，在完全相同的固定场景重复；不要只保留更好的一次结果。
10. 再执行完整的距离、倾角、图像位置和 XYZ 方向测试矩阵。
11. 使用独立 Ground Truth 重复实验，报告 Bias、MAE、RMSE 与 P95 Absolute Error。

验收记录至少应保存 `config.yaml` 副本、启动诊断日志、CSV、JSONL、bag、点云异常样本、Ground Truth 说明和测试环境照片。

## 15. IR → Depth 对齐的安全约束

“IR 和 Depth 数组 shape 相同”并不足以证明同一个 `(u,v)` 指向同一条相机射线。直接把未对齐的 IR 角点索引到 Depth，可能产生难以从 STD 中发现的系统误差。

本项目采用以下契约：

1. Z16 Depth 始终保留在原生 Depth viewport，不对深度图做 resize、crop 或二次重采样；
2. 使用 `rs.align(rs.stream.depth)` 把 Left IR1 映射到原生 Depth viewport；
3. 验证 align 后 Depth profile 仍与原生 Depth intrinsics 一致，并验证返回的 IR stream index 确实是 1；
4. QR 检测坐标、ROI mask、Depth pixel 和反投影全部使用这个 Depth viewport；
5. 反投影由 RealSense SDK 使用原生 Depth intrinsics 和畸变模型完成；
6. 如果 `rs.align` 不可用，只在 IR/Depth 内参相同且两者外参为单位旋转、零平移时允许直接 passthrough；
7. 上述严格验证失败就抛出 `FrameAlignmentError`，而不是仅凭分辨率相同继续测量。

不要通过 OpenCV `resize` 让两张图“看起来一样大”，也不要把 IR intrinsics 和 Depth pixels 混用。更换分辨率、固件、相机或 bag 后，应重新核对启动日志中的 `Pixel alignment` 与标定参数。

## 16. Plane RMS 与 RANSAC 阈值的重要警告

原始设计同时给出了 4 mm 的 RANSAC 内点阈值和 6 mm 的内点 RMS 上限。严格地说，这不是两个数值定义相互矛盾，而是让 6 mm gate 成为永远不会触发的死门控：本实现会在 SVD 精修后按最终平面重新分类内点；若最终内点残差为 `rᵢ`、RANSAC 阈值为 `τ=4 mm`，则每个 `rᵢ < τ`，因此 `RMS = sqrt(mean(rᵢ²)) < τ = 4 mm`。所以任何不小于 4 mm 的 RMS 上限都无法拒绝该定义下的最终内点平面。

本实现把默认 RMS 上限修正为 3 mm：

```yaml
plane:
  distance_threshold: 0.004  # 4 mm
  max_rms: 0.003             # 3 mm
```

配置校验会强制 `max_rms < distance_threshold`，保证 RMS gate 不是死门控。3 mm 只是可运行的工程起点，不是精度承诺；仍应通过固定 bag 和 Ground Truth 实验，根据工作距离、有效率与误差分布重新标定。不要为了放宽 RMS 而盲目放宽 RANSAC 内点阈值，这会把更多背景或飞点纳入平面。

另一种定义是对全部 ROI 点计算 RMS，但那会改变当前指标语义和门控逻辑，不能与现有实验结果直接比较。每次修改阈值都应保存配置、固定随机种子、回放相同 bag，并重新做 Ground Truth。

## 17. 常见错误排查

### `No module named pyrealsense2`

- 确认正在使用安装依赖时的同一个虚拟环境；
- 优先换用 Python 3.10～3.12；
- 查看官方 Python Wrapper 是否为当前 OS/CPU/Python 提供 wheel；
- 若无 wheel，按 librealsense 官方说明从源码构建绑定。

### 找不到设备、权限错误或启动超时

- 用 `realsense-viewer` 验证设备；
- 更换直连 USB 3.x 端口和合格线缆，避免无供电 Hub；
- Linux 安装 udev rules 后重新插拔设备；
- 检查是否有另一个进程占用相机；
- 多相机时在配置中填写正确的 serial number。

### 无法启动 `848x480@30` 的 Depth + IR1

- 检查设备确实是 D435i，而不是不兼容型号；
- 检查 USB 链路是否降级；
- 更新或回退到经验证的固件/SDK组合；
- 不要在未重新验证标定和测量误差的情况下随意改流规格。

### OpenCV 无法导入或窗口打不开

- 重装与 Python/平台匹配的 `opencv-python>=4.8,<5`；
- SSH/服务器使用 `--no-display`；
- 如果系统只有 headless OpenCV，不能使用 UI，但记录与计算仍可在 `--no-display` 下运行。

### `qr_not_found` / `qr_too_small`

- 确认 payload 精确为 `QR_A`、`QR_B`；
- 增大打印尺寸或移近相机；
- 改善均匀照明、对焦、对比度与 quiet zone；
- 默认 emitter 关闭，分别实测 ON/OFF，不要凭主观图像选择。

### `not_enough_depth_points`

- 查看 Depth 热力图和 point dump；
- 检查反光、透明、黑色吸收材料、遮挡与深度孔洞；
- 降低 `sample_stride` 会增加采样量但增加计算；
- 不要在没有误差实验时直接降低 `min_points`。

### `low_valid_depth_ratio` / `poor_spatial_support`

- 用 `--debug` 查看每个 QR 的 `Depth valid xx.x% Q=n/4`，并对照 Depth 热力图确认孔洞位置；
- `low_valid_depth_ratio` 表示 ROI 采样位置不少，但落在有效深度范围且成功反投影的比例过低，优先检查反光/透明/过黑材质、遮挡、工作距离和 IR→Depth 对齐；
- `poor_spatial_support` 在没有 Plane RMS 时通常是拟合前有效点分布不均；已有 Plane RMS/内点率时则是最终平面内点未覆盖足够象限；
- 优先改善安装姿态、表面与成像条件，或检查 ROI 是否越界。不要只为通过门控就降低 `min_valid_depth_ratio`、`min_spatial_quadrants` 或 `min_points_per_quadrant`。

### `deprojection_failed`

- 核对启动日志中的正数 depth scale、原生 Depth intrinsics、alignment strategy 与 IR/Depth shape；
- 检查 bag 是否保留了兼容标定、ROI 是否有效，以及当前 `pyrealsense2`/固件组合是否与录制环境兼容；
- 使用 `--debug --log-level DEBUG` 查看异常上下文。不要用 IR intrinsics 反投影 Depth pixel，也不要用 resize 掩盖 viewport 不一致。

### `low_inlier_ratio` / `ransac_failed`

- 检查 ROI 是否越过 QR 平面边缘或包含背景；
- 检查打印物是否翘曲；
- 检查 IR→Depth 对齐日志；
- 用 `--dump-points` 查看 `inlier` 分布，再调整 shrink ratio 或 RANSAC 参数。

### bag 无法录制或回放

- 录制文件必须以 `.bag` 结尾、父目录存在且目标文件不存在；
- 回放 bag 必须存在并包含 Depth 与 Left IR1；
- 正常 EOF 不是崩溃；若启动即 EOF，检查 bag 是否完整；
- `--repeat-bag` 会在每轮重新预热；若报告没有后预热帧，说明一轮长度不大于 `camera.warmup_frames`，应使用更长 bag 或在专用配置中减小预热数；
- 旧设备或不同流配置的 bag 可能被 D435i/标定契约拒绝。

### 帧采集超时 / `FrameAcquisitionError`

- 检查 USB 3.x 链路、供电、线缆占用、设备是否被其他程序打开，以及 `camera.frame_timeout_ms` 是否适合当前环境；
- 每次采集错误都会立即清空时序窗口，因为失败帧没有可信设备时间戳；恢复后 UI 显示 warming 属于预期行为；
- 连续错误达到程序上限会终止运行。不要把调大 timeout 当作修复 USB 掉线或坏 bag 的替代方案。

## 18. 当前验证边界与非目标

当前代码交付不等同于以下结论：

- 尚未在本开发环境连接真实 D435i 验证设备发现、emitter、30 FPS、长时间 USB 稳定性；
- 尚未实测 `rs.align` 在目标 SDK/固件组合上的 Left IR1 行为；
- 尚未验证 bag 录制文件可被目标机器稳定回放；
- 尚未完成真实 1000 帧运行、处理频率与内存稳定性测试；
- 尚未获得 Ground Truth Bias、MAE、RMSE 或 P95，因此不能承诺毫米级绝对精度；
- 文档中的预期误差量级和默认阈值都不是产品保证值。

第一版明确不包含 RGB 主链、PnP/Bundle Adjustment、IMU Fusion、Kalman Filter、SLAM、ROS、机器人坐标变换、手眼标定、多相机、神经网络、Web 前端或云服务。后续可以在 `P_camera` 之后增加外参变换，但不应修改已经验证的 QR、平面和射线核心语义。

## 19. 项目结构

```text
camera/         RealSense 真机、bag、帧源契约、对齐与设备诊断
detection/      多 QR 检测、payload 过滤、角点细化
geometry/       四边形、Depth 反投影、RANSAC、SVD、射线平面求交
measurement/    单 QR 三维定位、质量门控、双点距离
statistics/     有限窗口 Median/Mean/STD/MAD 与 stale 处理
visualization/  OpenCV IR/Depth debug UI
recording/      CSV、JSONL、point cloud dump
evaluation/     Ground Truth 指标与 Method A/B/C 比较
tools/          可复现的 QR_A / QR_B 生成工具
tests/          不依赖真实相机的自动化测试
utils/          配置加载和校验
main.py         CLI 与实时主循环
config.yaml     默认参数
pyproject.toml  项目元数据与唯一依赖声明
uv.lock         uv 跨平台锁文件，应随代码提交
.python-version uv 默认 Python 版本（3.11）
```

## 20. 许可证

本仓库当前未附带开源许可证。将代码上传到 GitHub 本身不等于授予复制、修改或再分发权利；如需开放给他人复用，应由项目所有者明确选择并添加 MIT、Apache-2.0、BSD-3-Clause 或其他合适许可证。
