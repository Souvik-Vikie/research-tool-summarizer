# Streamlit Cloud Deployment Guide

## Files Created for Deployment

1. **requirements_streamlit.txt** - Clean dependencies (no Windows-specific packages)
2. **runtime.txt** - Python version specification (3.11)
3. **.streamlit/config.toml** - Streamlit configuration

## Steps to Deploy

### 1. Update Your GitHub Repository

```bash
# Make sure you're in the project directory
cd c:\Users\User\Documents\Langchain\langchain\2_news_research_tool_project

# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Streamlit Cloud deployment"

# Push to GitHub (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/research-tool-summarizer.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/research-tool-summarizer`
   - **Branch**: `main`
   - **Main file path**: `main.py`
   - **Python version**: Will use `runtime.txt` (Python 3.11)
4. Click **"Advanced settings"**
5. In **"Python version"**, make sure it shows Python 3.11
6. In **"Requirements file"**, change to: `requirements_streamlit.txt`
7. Click **"Deploy!"**

### 3. Wait for Deployment

- Initial deployment takes 5-10 minutes (models need to download)
- You'll see the build logs
- Once complete, your app will be live!

### 4. Important Notes

**Memory Issues:**
- Free tier has 1GB RAM limit
- Your models (FLAN-T5 + embeddings) use ~800MB
- If you get memory errors, consider upgrading or using smaller models

**First Load:**
- First user will experience 1-2 minute load time (models downloading)
- Subsequent loads are faster (cached)

**Alternative: Use Hugging Face Spaces (Recommended)**

If Streamlit Cloud runs out of memory:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Select **"Streamlit"** SDK
4. Upload your files
5. Use `requirements_streamlit.txt`
6. Much better for ML models (16GB RAM free!)

## Troubleshooting

### Error: "python-magic-bin not found"
✅ **Fixed** - Removed from requirements_streamlit.txt

### Error: "Python version mismatch"
✅ **Fixed** - Added runtime.txt with Python 3.11

### Error: "Memory limit exceeded"
**Solution**: Deploy to Hugging Face Spaces instead (16GB RAM)

### Error: "Module not found"
**Check**: Make sure you're using `requirements_streamlit.txt` in Advanced Settings

## Your App URLs

After deployment:
- **Streamlit Cloud**: `https://YOUR_USERNAME-research-tool-summarizer.streamlit.app`
- **Hugging Face**: `https://huggingface.co/spaces/YOUR_USERNAME/research-tool`

## Testing Locally Before Deployment

```bash
# Test with deployment requirements
pip install -r requirements_streamlit.txt
streamlit run main.py
```

If it works locally with `requirements_streamlit.txt`, it should work on Streamlit Cloud!
