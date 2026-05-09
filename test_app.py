import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


# stub heavy / GUI deps so importing app.py doesn't pull torch or open a browser
streamlit = types.ModuleType("streamlit")
streamlit.cache_resource = lambda **k: (lambda f: f)
streamlit.set_page_config = lambda **k: None
sys.modules["streamlit"] = streamlit

transformers = types.ModuleType("transformers")
transformers.pipeline = lambda *a, **k: None
transformers.AutoModelForImageTextToText = object
transformers.AutoModelForSeq2SeqLM = object
transformers.AutoProcessor = object
transformers.AutoTokenizer = object
sys.modules["transformers"] = transformers

gtts = types.ModuleType("gtts")
gtts.gTTS = lambda **k: None
sys.modules["gtts"] = gtts

PIL = types.ModuleType("PIL")
PIL_Image = types.ModuleType("PIL.Image")
PIL_Image.Image = type("Image", (), {})
PIL_Image.open = lambda *a, **k: None
PIL.Image = PIL_Image
sys.modules["PIL"] = PIL
sys.modules["PIL.Image"] = PIL_Image


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app", root / "app.py")
app = importlib.util.module_from_spec(spec)
old_cwd = os.getcwd()
os.chdir(root)
try:
    spec.loader.exec_module(app)
finally:
    os.chdir(old_cwd)


class CleanStoryTests(unittest.TestCase):

    def assert_window(self, text):
        n = len(text.split())
        self.assertGreaterEqual(n, app.MIN_WORDS, f"too short ({n} words): {text!r}")
        self.assertLessEqual(n, app.MAX_WORDS, f"too long ({n} words): {text!r}")

    def test_short_input_is_padded(self):
        self.assert_window(app.clean_story("Once upon a time, a cat sat."))

    def test_long_input_is_truncated(self):
        self.assert_window(app.clean_story(" ".join(["word"] * 200)))

    def test_exact_length_is_preserved(self):
        out = app.clean_story(" ".join(["word"] * 60) + ".")
        self.assertEqual(60, len(out.split()))

    def test_empty_input_falls_back_to_default(self):
        out = app.clean_story("")
        self.assert_window(out)
        self.assertTrue(out.lower().startswith("once upon"))

    def test_inflected_unsafe_words_are_scrubbed(self):
        out = app.clean_story("The dragon was killed and his blood spilled. People died.")
        self.assert_window(out)
        for bad in ("kill", "blood", "die"):
            self.assertNotIn(bad, out.lower())

    def test_junk_markers_are_cut(self):
        out = app.clean_story("A dog ran. Story: chapter two http://x.com extra text")
        self.assert_window(out)
        self.assertNotIn("http", out.lower())
        self.assertNotIn("chapter", out.lower())

    def test_meta_storytail_is_cut(self):
        raw = (
            "Once upon a time, a kind little bear walked through a sunny "
            "forest with happy friends and they played all day long until "
            "the moon came up. If you like children's books, you'll love "
            "this story. Subscribe to read more."
        )
        out = app.clean_story(raw)
        self.assert_window(out)
        for tail in ("if you like", "subscribe", "you'll love"):
            self.assertNotIn(tail, out.lower())

    def test_prompt_leakage_is_cut(self):
        raw = (
            "Once upon a time, a happy bird sat in the green grass. It liked "
            "to hop and chirp and watch fluffy clouds drift by all morning. "
            "You are a kind children's book author. Write a short happy "
            "story for a 5-year-old child. Example -- Picture: A cat."
        )
        out = app.clean_story(raw)
        self.assert_window(out)
        for tell in ("you are a", "write a", "example", "picture:", "5-year-old"):
            self.assertNotIn(tell, out.lower())

    def test_result_ends_with_terminator(self):
        out = app.clean_story("a quick walk")
        self.assertIn(out[-1], ".!?")


class IsAcceptableTests(unittest.TestCase):

    def test_good_story_passes(self):
        self.assertTrue(app.is_acceptable(
            "Once upon a time, a kind little bear walked through a sunny "
            "forest with happy friends and they laughed, played and went home."
        ))

    def test_died_is_rejected(self):
        self.assertFalse(app.is_acceptable(
            "Once upon a time, his wife died on Christmas day."
        ))

    def test_killed_inflection_is_rejected(self):
        self.assertFalse(app.is_acceptable(
            "Once upon a time, the dragon killed everyone."
        ))

    def test_blank_marker_is_rejected(self):
        self.assertFalse(app.is_acceptable("____ was here today afterwards."))

    def test_url_is_rejected(self):
        self.assertFalse(app.is_acceptable(
            "Once upon a time, visit http://example.com to learn more please."
        ))

    def test_too_short_is_rejected(self):
        self.assertFalse(app.is_acceptable("Once upon a time."))

    def test_evil_demon_is_rejected(self):
        self.assertFalse(app.is_acceptable(
            "An evil demon laughed loudly in the dark scary cave forever."
        ))

    def test_empty_is_rejected(self):
        self.assertFalse(app.is_acceptable(""))


class SafeTemplateTests(unittest.TestCase):

    def test_template_passes_safety_and_length(self):
        captions = [
            "A cat sitting on a mat",
            "the colourful sky above the trees",
            "two friends playing in a sunny park",
            "An apple on a wooden table",
        ]
        for caption in captions:
            with self.subTest(caption=caption):
                story = app.safe_template(caption)
                cleaned = app.clean_story(story)
                self.assertTrue(app.is_acceptable(cleaned))
                n = len(cleaned.split())
                self.assertGreaterEqual(n, app.MIN_WORDS)
                self.assertLessEqual(n, app.MAX_WORDS)

    def test_template_starts_with_storybook_phrase(self):
        story = app.safe_template("a fluffy puppy")
        self.assertTrue(story.lower().startswith("once upon a time"))


class BuildPromptTests(unittest.TestCase):

    def test_prompt_contains_caption(self):
        self.assertIn("A puppy in a basket", app.build_prompt("A puppy in a basket"))

    def test_prompt_demands_child_friendly_tone(self):
        prompt = app.build_prompt("X").lower()
        self.assertIn("happy", prompt)
        self.assertIn("child", prompt)

    def test_prompt_includes_few_shot_example(self):
        prompt = app.build_prompt("anything")
        self.assertIn("Once upon a time", prompt)
        self.assertIn("Mimi", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
