# Custom-Trained-Bengali-Voice-Cloning-Model-With-Google-Colab

Building and publishing your own custom-trained Bengali voice cloning model and hosting it on Hugging Face for free public access involves a structured four-phase pipeline: **Dataset Preparation**, **Training on Google Colab (using a GPU)**, **Uploading to Hugging Face**, and **Deploying a Free Web Demo (Hugging Face Space)**.

Open-source architectures like **Coqui XTTS-v2** or **IndicF5 / F5-TTS** are ideal for this because they support multi-lingual support, fine-tuning, and zero-shot cloning.
[ehzawad/indicf5-bangla-tts](https://huggingface.co/ehzawad/indicf5-bangla-tts#:~:text=Collection%20including%20ehzawad%2Findicf5%2Dbangla%2Dtts.%20Bangla%20Text%2Dto%2DSpeech%20(TTS)%20Collection.,the%20conditions%20to%20access%20the%20files%20info.)

---

### Phase 1: Prepare Your Bengali Dataset

To train a clean voice model, you need a dataset structured in standard **LJSpeech format**.

1. **Record Audio:** Record 10 to 30 minutes of clean, high-quality audio of you speaking Bengali clearly. Avoid background noise, echo, or heavy compression.
2. **Slice Clips:** Cut the recordings into short segments (3 to 10 seconds long) and export them as `.wav` files (Mono, 22050 Hz or 24000 Hz sample rate).
3. **Organize Directory Structure:** Create a folder structure in your Google Drive or Colab workspace:
```text
bengali_dataset/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
└── metadata.csv

```


4. **Format `metadata.csv`:** Inside the CSV file, map each audio filename to its exact Bengali text transcript using a pipe (`|`) delimiter:
```text
audio_001|আমি বাংলা ভাষায় কথা বলছি|ami bangla vashay kotha bolchi
audio_002|এটি আমার কাস্টম ভয়েস ক্লোনিং মডেল|eti amar custom voice cloning model

```



---

### Phase 2: Train the Model on Google Colab (Free T4 GPU)

Open a new Google Colab notebook, change your runtime type to **T4 GPU**, and execute the following steps using an established open-source fine-tuning wrapper (such as an XTTS or F5-TTS training script).

#### 1. Install Dependencies & Clone a Trainer Repo

```python
# Mount Google Drive if your dataset is stored there
from google.colab import drive
drive.mount('/content/drive')

# Install Coqui TTS or required training libraries
!pip install TTS
!pip install huggingface_hub

```

#### 2. Configure Training Settings

If using Coqui XTTS-v2 (which handles multi-lingual weights and voice cloning exceptionally well), configure your training run:

```python
import torch
from TTS.api import TTS

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# You can use community-proven notebooks or run training via the TTS CLI/API wrapper
# Example configuration snippet for training initiation:
!tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --train_data_dir /content/drive/MyDrive/bengali_dataset \
    --language_idx bn \
    --output_path /content/drive/MyDrive/bengali_output

```

*Note: Depending on your dataset size, training can take anywhere from 30 minutes to a couple of hours on a Colab T4 GPU.*

---

### Phase 3: Upload Your Trained Model to Hugging Face Hub

Once training completes, you will find your model checkpoint files (typically `model.safetensors` or `model.pth`, `config.json`, and `vocab.json`) inside your output directory. You can push them directly to Hugging Face from Colab.

#### 1. Authenticate with Hugging Face

```python
from huggingface_hub import notebook_login, HfApi

# This will prompt you to enter your Hugging Face Write Token
notebook_login()

```

#### 2. Create Repository and Upload Files

```python
api = HfApi()
repo_id = "your-username/bengali-custom-voice-model"  # Replace with your HF username and desired repo name

# Create model repository on Hugging Face
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Upload your trained model directory
api.upload_folder(
    folder_path="/content/drive/MyDrive/bengali_output/run/training/best_model",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model successfully uploaded to https://huggingface.co/{repo_id}")

```

---

### Phase 4: Create a Free Web Demo (Hugging Face Space)

To allow other Bengali users to use your model for free through a clean user interface, set up a **Hugging Face Space** running **Gradio**.

1. Go to Hugging Face and click **New Space**.
2. Name your Space, select **Gradio** as the Space SDK, and choose **Public**.
3. Create an `app.py` file inside your Space repository with a script that pulls your trained model and sets up an inference UI:

```python
import gradio as gr
from TTS.api import TTS
import torch

# Load your custom fine-tuned model from Hugging Face
model_name = "your-username/bengali-custom-voice-model"
tts = TTS(model_name=model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def synthesize_bengali(text, reference_audio):
    output_path = "output.wav"
    # Run voice generation / cloning inference
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_audio,
        language="bn"
    )
    return output_path

# Build Gradio Web Interface
demo = gr.Interface(
    fn=synthesize_bengali,
    inputs=[
        gr.Textbox(label="Enter Bengali Text (বাংলা টেক্সট লিখুন)", value="বাংলাদেশ একটি সুন্দর দেশ।"),
        gr.Audio(label="Reference Voice Audio (Sample)", type="filepath")
    ],
    outputs=gr.Audio(label="Generated Cloned Voice"),
    title="Custom Bengali Voice Cloning",
    description="Type Bengali text and provide a reference sample to generate speech using the custom-trained model."
)

if __name__ == "__main__":
    demo.launch()

```

4. Add a `requirements.txt` file to your Space containing:
```text
torch
TTS
gradio
transformers

```


5. Hugging Face Spaces will automatically build and host your web app for free on a CPU tier (or you can upgrade to a community GPU tier if needed), allowing anyone in the Bengali community to access, test, and use your voice model directly in their browser.
