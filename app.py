import io
import os
import re

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-base"

MIN_WORDS = 50
MAX_WORDS = 100

UNSAFE_WORDS = (
    "kill", "murder", "blood", "gun", "weapon", "drug", "drunk",
    "hate", "stupid", "idiot", "die", "dead", "death",
    "sex", "nude", "naked",
    "horror", "demon", "devil", "evil",
    "suicide", "abuse", "rape",
    "hurt", "wound", "bleed", "injur", "sick", "illness",
    "vet", "hospital", "damag", "broken", "crash",
)


st.set_page_config(
    page_title="Magic Story Maker",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the picture-reader robot ...")
def load_captioner():
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL)
    return pipeline(
        "image-to-text",
        model=model,
        image_processor=processor,
        tokenizer=processor.tokenizer,
    )


@st.cache_resource(show_spinner="Loading the story-writer robot ...")
def load_story_generator():
    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(STORY_MODEL)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def caption_image(image):
    captioner = load_captioner()
    text = captioner(image)[0]["generated_text"].strip()
    if not text:
        return "A lovely scene"
    return text[0].upper() + text[1:]


def build_prompt(caption):
    return (
        "You are a kind children's book author. Write a short happy bedtime "
        "story for a 5-year-old child. The story must be cheerful from start "
        "to end, use simple words, have friendly characters, and finish with "
        "everyone safe, smiling and going home happily. Never mention illness, "
        "injury, sadness, fear, or anything scary.\n\n"
        "Example -- Picture: A cat sitting on a sofa.\n"
        "Story: Once upon a time, a fluffy little cat named Mimi sat on a soft "
        "sunny sofa. She purred a happy tune while watching butterflies dance "
        "outside the window. Her best friend, a tiny mouse named Toot, came "
        "over to share a juicy strawberry. They giggled, hugged warmly, and "
        "watched fluffy clouds make funny shapes until the moon whispered "
        "good night.\n\n"
        f"Now write a similar happy story. Picture: {caption}.\nStory:"
    )


def safe_template(caption):
    subject = caption.strip().rstrip(".").lower()
    if subject.startswith(("a ", "an ", "the ")):
        phrase = subject
    else:
        phrase = "a wonderful little friend"
    return (
        f"Once upon a time, there was {phrase}. "
        "It looked happy and bright in the warm golden sunshine. "
        "A kind little bunny came hopping along to say a cheerful hello. "
        "They smiled at each other, played gentle games on the soft grass, "
        "and shared sweet juicy berries together under the blue sky. "
        "The wind sang a soft song, and tiny stars began to twinkle high above. "
        "Then they laughed, hugged warmly, and went home with happy hearts, "
        "dreaming of new adventures waiting for them tomorrow."
    )


def is_acceptable(text):
    if not text or len(text.split()) < 12:
        return False
    lower = text.lower()
    for bad in ("____", "http", "www.", "@"):
        if bad in lower:
            return False
    for word in UNSAFE_WORDS:
        if re.search(rf"\b{re.escape(word)}\w*", lower):
            return False
    for tell in ("i don't know", "chapter", "section", "please", "click"):
        if tell in lower:
            return False
    return True


def clean_story(text):
    text = re.sub(r"\s+", " ", text).strip()

    cut_markers = (
        "Story:", "\n\n", "http", "www.", "Chapter", "###",
        "If you like", "If you enjoy", "If you want to read",
        "Subscribe", "Click here", "Read more",
        "you'll love", "you will love", "this story",
        "You are a", "Write a", "Tell a story",
        "Example --", "Example —", "Picture:",
        "This is a happy", "This is a short",
        "for a 5-year-old", "children's book author",
    )
    for marker in cut_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()

    for word in UNSAFE_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\w*", "happy", text, flags=re.IGNORECASE)

    words = text.split()
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]
        text = " ".join(words)
        for stop in (".", "!", "?"):
            idx = text.rfind(stop)
            if idx >= len(text) * 0.6:
                text = text[: idx + 1]
                break

    fillers = [
        "They laughed together, hugged each other, and felt very happy inside.",
        "The sky turned soft pink, and a gentle wind sang a tiny lullaby.",
        "Little stars peeked through the clouds, blinking like sleepy eyes.",
        "A friendly bird chirped a sweet tune from the tallest green tree.",
        "Everyone shared warm cookies and giggled at the funniest jokes.",
        "Then they walked home with smiles, dreaming of tomorrow's adventure.",
    ]
    if not text:
        text = "Once upon a time, a kind little friend went on a tiny adventure."
    i = 0
    while len(text.split()) < MIN_WORDS and i < 20:
        text = text.rstrip(" .!?") + ". " + fillers[i % len(fillers)]
        i += 1

    if text and text[-1] not in ".!?":
        text += "."

    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]).rstrip(" .!?") + "."

    return text


def generate_story(caption):
    generator = load_story_generator()
    output = generator(
        build_prompt(caption),
        max_new_tokens=160,
        min_new_tokens=70,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )[0]["generated_text"]

    story = output.strip()
    if not story.lower().startswith("once upon a time"):
        story = "Once upon a time, " + story

    if not is_acceptable(story):
        story = safe_template(caption)

    return clean_story(story)


def make_audio(text):
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def main():
    with st.sidebar:
        st.header("How it works 🪄")
        st.markdown(
            "1. **Upload** any picture you like.\n"
            "2. The **picture-reader robot** looks at it.\n"
            "3. The **story-writer robot** writes a tiny tale.\n"
            "4. Press **Play** and listen!\n"
        )
        st.divider()
        st.caption(
            "Models used:\n"
            f"- `{CAPTION_MODEL}`\n"
            f"- `{STORY_MODEL}`\n"
            "- gTTS (Google Text-to-Speech)"
        )
        st.caption("Made for kids aged 3-10.")

    st.title("📖 Magic Story Maker")
    st.caption(
        "Upload a picture and a friendly robot will write a tiny story "
        "just for you, then read it out loud! 🦄✨"
    )

    uploaded = st.file_uploader(
        "Choose a picture (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("⬆️ Please upload an image to begin.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your picture", width="stretch")

    if not st.button("✨ Make my story!", type="primary", width="stretch"):
        return

    try:
        with st.spinner("👀 Looking at your picture..."):
            caption = caption_image(image)
        st.success(f"I see: **{caption}**")

        with st.spinner("✍️ Writing a magical story..."):
            story = generate_story(caption)
        st.subheader("📚 Your Story")
        st.write(story)
        st.caption(f"({len(story.split())} words)")

        with st.spinner("🎙️ Recording the voice..."):
            audio = make_audio(story)
        st.subheader("🎧 Listen")
        st.audio(audio, format="audio/mp3")

        st.download_button(
            label="⬇️ Download story audio",
            data=audio,
            file_name="story.mp3",
            mime="audio/mp3",
            width="stretch",
        )
    except Exception as e:
        st.error(
            "Oh no! Something went wrong while making your story. "
            "Please try a different picture."
        )
        st.exception(e)


if __name__ == "__main__":
    main()
