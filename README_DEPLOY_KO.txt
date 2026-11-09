Municipal Bond Cost Prediction Model - Render public deployment

1. This folder is a deployment-only copy of the research website. The original numerical models, HTML, and embedded chain calculation engine are unchanged. The bulky research CSV/Excel files and offline launcher scripts were omitted because the public server does not use them. Keep the original full research ZIP separately.

2. Go to https://github.com/new. Create a PRIVATE repository, e.g., municipal-bond-cost-predictor. Do not create a README in GitHub when making the empty repository. Upload the CONTENTS of this folder (all files and three asset/model directories), not the ZIP file. Keep render.yaml and .python-version at the repository root.

3. Go to https://dashboard.render.com/. Connect your GitHub account. Choose New > Blueprint, select the new private repository and approve the service configuration in render.yaml. This config selects Render Free by default, so review the cost before approving.

4. After build and deploy finish, copy the https://...onrender.com URL from the service page. Check both / and /api/health and send the URL to your professor.

5. Render Free has 512MB RAM and sleeps after 15 minutes of inactivity; first visit after sleep can take about one minute. If memory limit is exceeded or performance is insufficient, upgrade the compute plan. Do not assume the free tier is adequate until tested remotely.

6. Important: A private GitHub repo keeps the source code out of public GitHub, but the website and /api/predict endpoint are publicly accessible to anyone with the URL. This prototype has no user login or rate limit. If access must be restricted to specific professors, add authentication/access-control first.

The deployment uses Python 3.12 to avoid incompatible default Python versions. Build: pip install -r requirements.txt; Start: uvicorn app:app --host 0.0.0.0 --port $PORT.
