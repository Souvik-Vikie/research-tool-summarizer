# Deploy to Hugging Face Spaces (Git Method)

## Step 1: Install Git LFS

Hugging Face uses Git LFS for large files.

```bash
# Download from: https://git-lfs.github.com/
# Or if you have choco (Windows):
choco install git-lfs

# Initialize
git lfs install
```

## Step 2: Create Space on Hugging Face

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: `research-tool-summarizer`
3. SDK: **Streamlit**
4. Hardware: **CPU basic** (free)
5. Click **Create Space**

## Step 3: Prepare Your Files

```bash
cd c:\Users\User\Documents\Langchain\langchain\2_news_research_tool_project

# Rename requirements for deployment
copy requirements_streamlit.txt requirements.txt
```

## Step 4: Connect and Push to Hugging Face

```bash
# Add Hugging Face remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/research-tool-summarizer

# Create a clean branch for deployment
git checkout -b hf-deploy

# Add only necessary files
git add main.py requirements.txt .streamlit/

# Commit
git commit -m "Deploy to Hugging Face Spaces"

# Push to Hugging Face
git push hf hf-deploy:main
```

## Step 5: Access Your App

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/research-tool-summarizer
```

You can also embed it:
```
https://YOUR_USERNAME-research-tool-summarizer.hf.space
```

## Important Files for Hugging Face

**Required:**
- `main.py` - Your Streamlit app
- `requirements.txt` - Dependencies

**Optional:**
- `README.md` - Space description
- `.streamlit/config.toml` - Streamlit config

## Adding Secrets (If Needed)

If you have API keys or secrets:

1. Go to your Space settings
2. Click **"Settings"** → **"Repository secrets"**
3. Add your secrets (like API keys)
4. Access in code: `os.getenv("SECRET_NAME")`

## Troubleshooting

### Build fails with memory error
- Upgrade to CPU upgrade (still free but requires credit card)
- Or optimize model loading

### Models take too long to download
- This is normal on first build (~5-10 minutes)
- Subsequent builds are cached and faster

### App shows "Building..."
- Check the **Logs** tab for errors
- Make sure `requirements.txt` has all dependencies

## Your Resume Link

Add this to your resume:
```
🔗 Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/research-tool-summarizer
```

Or use the shorter URL:
```
🔗 https://YOUR_USERNAME-research-tool-summarizer.hf.space
```
