# Push to GitHub

The files are committed locally. Here's how to push to GitHub:

## Option 1: Using Existing Repository

If you already have a GitHub repo:

```bash
cd /Users/hehe/worksplace/tennis_coach

# Add your GitHub repo as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub details.

## Option 2: Create New Repository

1. Go to https://github.com/new
2. Create a new repository called `tennis_coach`
3. Run:

```bash
cd /Users/hehe/worksplace/tennis_coach

git remote add origin https://github.com/YOUR_USERNAME/tennis_coach.git
git branch -M main
git push -u origin main
```

## Direct Colab Link

Once pushed to GitHub, use this URL format in Colab:

```
https://github.com/YOUR_USERNAME/tennis_coach/blob/main/Tennis_Ball_Tracking_GPU.ipynb
```

Then open it with: https://colab.research.google.com/github/YOUR_USERNAME/tennis_coach/blob/main/Tennis_Ball_Tracking_GPU.ipynb

## Files Available

- **Tennis_Ball_Tracking_GPU.ipynb** - Colab notebook (GPU accelerated)
- **COLAB_SETUP_GUIDE.md** - Setup instructions
- **ball_tracking_only.py** - Local simplified script
- **run_ball_tracking.py** - Local script with progress tracking
