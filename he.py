python -m venv .venv
.venv\Scripts\activate


python -m pip install --upgrade pip
pip install "numpy<2.0" scipy==1.13.1
pip install torch==2.2.2+cpu torchaudio==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 sounddevice==0.4.7 soundfile==0.13.1 tqdm==4.67.1
pip install audb  # if you want EmoDB


Download from: https://www.gyan.dev/ffmpeg/builds/

(pick release full → unzip somewhere, e.g., C:\ffmpeg)

Add C:\ffmpeg\bin to your PATH:

Search Environment Variables → Edit environment variables

Edit Path → Add C:\ffmpeg\bin

Restart PyCharm / terminal
