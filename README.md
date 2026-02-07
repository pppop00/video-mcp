# video-mcp first frame batch generator

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

## 3. 常用参数

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
