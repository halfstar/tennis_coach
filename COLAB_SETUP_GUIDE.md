# Google Colab Setup Guide - Tennis Ball Tracking with GPU

## Quick Start

You now have everything set up for fast GPU-accelerated ball tracking! Here's how to use it:

### Step 1: Prepare Your Google Drive

1. **Create a folder** in Google Drive called `tennis_coach`
2. **Upload these files** to that folder:
   - `test_video_1280x720.mp4` (your video file)
   - Create a `models` subfolder and upload `tracknet_model.pt` inside it
   
Your Drive structure should look like:
```
My Drive/
└── tennis_coach/
    ├── test_video_1280x720.mp4
    └── models/
        └── tracknet_model.pt
```

### Step 2: Open Google Colab

1. Go to https://colab.research.google.com
2. Click **File → Open Notebook → Upload**
3. Upload the `Tennis_Ball_Tracking_GPU.ipynb` file from this folder

### Step 3: Configure GPU

1. Click **Runtime → Change Runtime Type**
2. Select **GPU** as the accelerator
3. Click **Save**

### Step 4: Run the Notebook

Execute each cell in order:

1. **Cell 1**: Install dependencies
2. **Cell 2**: Check GPU and imports
3. **Cell 3**: Mount Google Drive
4. **Cell 4-6**: Define the model and detector
5. **Cell 7**: Load your video
6. **Cell 8**: Run ball detection ⚡ (FAST on GPU!)
7. **Cell 9**: Generate output video
8. **Cell 10**: Download results
9. **Cell 11**: (Optional) Export tracking data as CSV

## Expected Performance

| Component | CPU (your laptop) | GPU (Colab) |
|-----------|------------------|-----------|
| Ball Detection | ~70 sec/frame | ~0.5-1 sec/frame |
| For 10-min video (18,278 frames) | **~35 hours** | **3-6 hours** |
| **Total Runtime** | **~50+ hours** | **~4-8 hours** |

## What You'll Get

- **output_ball_tracking_gpu.mp4**: Video with green circles marking detected ball positions
- **ball_tracking_data.csv**: Frame-by-frame X,Y coordinates of detected balls

## Troubleshooting

### Error: "Model not found"
- Make sure you uploaded `tracknet_model.pt` to the `models` folder in Drive
- Check that the path matches: `models/tracknet_model.pt`

### Error: "Video not found"
- Make sure `test_video_1280x720.mp4` is in the root of `tennis_coach` folder
- Check file name spelling

### Out of Memory
- Try processing shorter video clips
- Or reduce frame resolution in the code (change `self.width` and `self.height`)

### Slow Processing
- Verify GPU is active: Runtime → Change Runtime Type → confirm GPU is selected
- Check cell 2 output shows "CUDA available: True"

## Performance Tips

1. **Use Colab Pro** for longer GPU access (25 hours/week → unlimited)
2. **Process in batches**: Split long videos into 1-2 minute clips
3. **Use frame skipping**: Modify to process every 2nd or 3rd frame if needed

## Local Alternative (if you prefer CPU)

You can also run locally using:
```bash
source tennis_env/bin/activate
python ball_tracking_only.py \
  --path_ball_track_model models/tracknet_model.pt \
  --path_input_video test_video_1280x720.mp4 \
  --path_output_video output_ball_tracking_local.mp4
```

## Next Steps After Ball Tracking

Once you have the ball tracking results, you can:
- Add court detection (detects court keypoints)
- Add player detection (tracks players)
- Combine with bounce detection
- Create advanced visualizations

Let me know if you need any of those features added!
