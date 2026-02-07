# video-mcp image tools

当前项目包含两个脚本：

- `batch_make_first_frame.py`：基于输入图片批量生成首帧图。
- `generate_video_dataset_from_images.py`：基于输入图片批量生成多帧样本与 `metadata.jsonl`。

批量读取 PNG/JPG/JPEG 图片，按你提供的版式生成首帧图，并输出为：

- `destination/<原图文件名>_first_frame_<MM-DD-YY-HHMMSS>.png`

所有加工结果都在同一个目录里，文件名保留原图 `stem`，并追加生成时的时间戳。

## 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 2. 批量生成

```bash
python3 batch_make_first_frame.py \
  --input-dir /path/to/image_folder \
  --output-dir /path/to/destination \
  --question "What is shown in the image?" \
  --option-a "A text" \
  --option-b "B text" \
  --option-c "C text" \
  --option-d "D text" \
  --workers 8
```

## 3. 首帧脚本常用参数

- `--recursive`：递归读取子目录中的 PNG/JPG/JPEG。
- `--width/--height`：输出分辨率，默认 `1920x1080`。
- `--font-path`：指定字体文件（建议中文场景传入中文字体）。
- `--workers`：并行处理线程数（1000 张图建议 8~16 试跑）。

## 4. 示例（你的结构）

```bash
python3 batch_make_first_frame.py \
  --input-dir /data/my_images \
  --output-dir /data/first_frames \
  --question "这张图的主要物体是什么？" \
  --option-a "时钟" \
  --option-b "猫头鹰" \
  --option-c "蜡烛" \
  --option-d "其他" \
  --workers 8
```

生成后示例：

- `/data/first_frames/image1_first_frame_02-07-26-002824.png`
- `/data/first_frames/cat_first_frame_02-07-26-002825.png`

## 5. 基于现有图片数据集生成多帧样本 + metadata.jsonl

如果你不想“从零合成场景”，而是要基于已有图片数据集生成视频帧序列，可用：

```bash
python3 generate_video_dataset_from_images.py \
  --input-dir /data/my_images \
  --output-dir /data/video_dataset \
  --split train \
  --question "这张图的主要物体是什么？" \
  --option-a "时钟" \
  --option-b "猫头鹰" \
  --option-c "蜡烛" \
  --option-d "其他" \
  --answer B \
  --fps 16 \
  --seconds 3 \
  --width 1024 \
  --height 768 \
  --overlay-mode all_frames \
  --workers 8 \
  --recursive
```

输出结构示例：

- `/data/video_dataset/train/000000/frames/frame_0000.png`
- `/data/video_dataset/train/000000/frames/frame_0001.png`
- `/data/video_dataset/train/metadata.jsonl`

`metadata.jsonl` 每行包含：

- `sample_id`
- `split`
- `source_image`
- `width/height/fps/seconds/num_frames`
- `question`
- `choices`
- `answer`
- `overlay_mode`
- `highlight_start/highlight_end`
- `frames_dir`

## 6. 多帧脚本常用参数

- `--overlay-mode`：`first_frame_only` 或 `all_frames`，控制题面显示帧范围。
- `--highlight-start/--highlight-end`：答案框高亮帧区间；`--highlight-end` 默认最后一帧。
- `--answer`：当前一次运行内所有 sample 共用同一个正确答案（A/B/C/D）。
- `--max-samples`：只处理前 N 张图，适合小规模试跑。
- `--seed`：控制每个 sample 的轻微镜头运动参数，便于复现结果。
- `--recursive`：递归读取子目录中的 PNG/JPG/JPEG。

## 7. 行为说明

- 多帧脚本输出目录固定为 `output-dir/split/sample_id/frames/frame_XXXX.png`。
- `sample_id` 为从 `000000` 开始的顺序编号。
- `metadata.jsonl` 位于 `output-dir/split/metadata.jsonl`，每行对应一个 sample。
