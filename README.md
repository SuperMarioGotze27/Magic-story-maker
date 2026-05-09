# Magic Story Maker

A small Streamlit app that turns a picture into a 50-100 word bedtime
story for kids and reads it out loud.

The pipeline runs three stages: BLIP describes the image, FLAN-T5 turns
that caption into a short story, and gTTS synthesises the audio.

## Stack

- `Salesforce/blip-image-captioning-base` for image captioning
- `google/flan-t5-base` for story generation
- `gTTS` (Google Text-to-Speech) for audio
- `streamlit` for the UI

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the two Hugging Face models into the local
cache; later runs are fast.

## Deploy on Streamlit Cloud

1. Push this folder to a public GitHub repo.
2. Open <https://share.streamlit.io>, click *New app*, point it at the
   repo, branch and `app.py`.
3. Click *Deploy*. Streamlit Cloud reads `requirements.txt` and
   `runtime.txt` automatically.
4. The first build takes about 3-5 minutes (model downloads).
5. Submit the resulting URL with the assignment.

If the free tier runs out of memory, swap `STORY_MODEL` in `app.py` for
`google/flan-t5-small`.

## Tests

The pure helpers in `app.py` (`clean_story`, `is_acceptable`,
`safe_template`, `build_prompt`) are covered by 22 unit tests using only
the standard library. Streamlit / transformers / gTTS / PIL are stubbed
out so no model is downloaded.

```bash
python -m unittest discover -s tests -v
```

The expected output ends with `Ran 22 tests in 0.0xxs / OK`.

Manual checks worth doing once before submission:

- Upload a JPG or PNG and confirm the caption looks reasonable.
- Click *Make my story!* and confirm the story word count is between 50
  and 100.
- Confirm the audio player loads and plays back the story.
- Use the download button to save `story.mp3`.

## Layout

```
app.py                  # Streamlit application
requirements.txt        # Python dependencies
runtime.txt             # Python 3.11 for Streamlit Cloud
.streamlit/config.toml  # Theme
tests/test_app.py       # Unit tests
docs/                   # 中文设计与部署文档
```
