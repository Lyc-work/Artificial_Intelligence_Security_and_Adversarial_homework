# 果壳人工智能安全与对抗图像生成赛道作业

本项目用于对原始视频进行预处理，提取目标帧与候选源帧，并根据质量、身份相似度、表情、姿态和光照等指标，为每个 target 自动选择最合适的 source 图片，最后调用 FaceFusion 完成批量换脸生成。

## 项目目录结构

```text
.
├── data
│   ├── videos/                  # 存放原始视频
│   └── image_list.txt           # 视频与人物 ID 列表
├── scripts
│   ├── prepare_generation_images.py
│   ├── select_source_per_target.py
│   └── batch_facefusion_source_map.py
├── work
│   ├── target_frames/           # 从视频中提取的 target 图片
│   ├── source_candidates/       # 从视频中抽取的 source 候选帧
│   ├── selected_sources_per_target/
│   └── logs/                    # source 筛选记录
└── facefusion/                  # FaceFusion 项目目录
```

## 1. 下载原始视频

将原始视频下载并放入以下目录：

```bash
./data/videos
```

请确保 `data/image_list.txt` 中记录的视频文件名、人物 ID 等信息与 `data/videos` 中的视频文件对应。

## 2. 安装 FaceFusion

请先将 FaceFusion 安装或克隆到项目目录下：

```text
./facefusion
```

安装完成后，应保证以下脚本路径存在：

```text
facefusion/facefusion.py
```

同时请根据 FaceFusion 官方要求配置好 Python 环境、模型文件和相关依赖。

## 3. 提取 target 图片和 source 候选帧

从视频中提取 target 图片，并从视频中提取 source 的候选帧图片。

其中，source 候选帧按照每个 ID 每隔 5 帧抽取一张图片。

```bash
python -m scripts.prepare_generation_images \
  --mode extract-all \
  --image-list data/image_list.txt \
  --videos-dir data/videos \
  --target-frames-dir work/target_frames \
  --source-candidates-dir work/source_candidates
```

运行后将生成：

```text
work/target_frames
work/source_candidates
```

## 4. 为每个 target 选择最合适的 source 图片

针对候选帧图片，自动选取最适用于 target 的 source 帧图片。

选择时综合考虑以下因素：

- 图像质量
- 身份相似度
- 表情匹配程度
- 姿态匹配程度
- 光照匹配程度

运行命令：

```bash
python -m scripts.select_source_per_target \
  --image-list data/image_list.txt \
  --target-dir work/target_frames \
  --source-candidates-dir work/source_candidates \
  --selected-source-dir work/selected_sources_per_target \
  --log-dir work/logs/source_per_target_selection \
  --w-quality 0.45 \
  --w-identity 0.25 \
  --w-expression 0.15 \
  --w-pose 0.10 \
  --w-lighting 0.05 \
  --overwrite
```

运行完成后，会在日志目录下生成 source 与 target 的匹配关系文件：

```text
work/logs/source_per_target_selection/selected_source_map.csv
```

该文件将用于后续批量调用 FaceFusion。

## 5. 调用 FaceFusion 批量生成换脸结果

使用上一步生成的 `selected_source_map.csv`，调用 FaceFusion 将 source 图片换到 target 图片上。

```bash
python -m scripts.batch_facefusion_source_map \
  --facefusion-script facefusion/facefusion.py \
  --source-map work/logs/source_per_target_selection/selected_source_map.csv \
  --target-dir work/target_frames \
  --output-dir work/facefusion_outputs
```

生成结果将保存到：

```text
work/facefusion_outputs
```

## 完整运行流程

```bash
python -m scripts.prepare_generation_images \
  --mode extract-all \
  --image-list data/image_list.txt \
  --videos-dir data/videos \
  --target-frames-dir work/target_frames \
  --source-candidates-dir work/source_candidates

python -m scripts.select_source_per_target \
  --image-list data/image_list.txt \
  --target-dir work/target_frames \
  --source-candidates-dir work/source_candidates \
  --selected-source-dir work/selected_sources_per_target \
  --log-dir work/logs/source_per_target_selection \
  --w-quality 0.45 \
  --w-identity 0.25 \
  --w-expression 0.15 \
  --w-pose 0.10 \
  --w-lighting 0.05 \
  --overwrite

python -m scripts.batch_facefusion_source_map \
  --facefusion-script facefusion/facefusion.py \
  --source-map work/logs/source_per_target_selection/selected_source_map.csv \
  --target-dir work/target_frames \
  --output-dir work/facefusion_outputs
```

## 注意事项

1. 请确保原始视频已经放入 `data/videos` 目录。
2. 请确保 `data/image_list.txt` 中的视频路径和 ID 信息正确。
3. 请确保 FaceFusion 已正确安装，并且 `facefusion/facefusion.py` 可以正常运行。
4. 如果更改了 `target_frames`、`source_candidates` 或日志目录路径，需要同步修改后续命令中的路径。
5. 本项目仅用于课程学习、科研实验或授权数据处理，请勿用于侵犯他人隐私、伪造身份、欺骗传播或其他违法违规用途。
