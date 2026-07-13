# Smart AI Meeting Assistant

Upload a meeting recording (or paste a transcript) and get back a summary, key decisions, and a list of action items with assignees and deadlines.

Transcription runs locally, only the text gets sent out for summarization.

## Stack
- Streamlit (UI)
- faster-whisper (local speech-to-text)
- Google Gemini API (google-genai SDK) for the summarization
- Pydantic for the structured output schema

## Setup

```
pip install streamlit google-genai pydantic python-dotenv faster-whisper
```

Add a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run it:
```
streamlit run main2.py
```

## Usage

Pick an input method, upload audio or paste text, hit process. Audio gets transcribed with Whisper first, then the transcript goes to Gemini for the summary/decisions/action items.
