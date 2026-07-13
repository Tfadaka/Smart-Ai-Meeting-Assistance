import os
import time
from google.genai import types
import streamlit as st
from google import genai
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Load API credentials
load_dotenv()

# Initialize Gemini Client
client = genai.Client()

# ==========================================
# DATA SCHEMA STRUCTURE
# ==========================================
class ActionItem(BaseModel):
    task: str = Field(description="The specific task that needs to be done.")
    assignee: str = Field(description="The person responsible for this task. Use 'Unassigned' if unknown.")
    deadline: str = Field(description="The mentioned deadline, or 'Not specified'.")

class MeetingSummary(BaseModel):
    title: str = Field(description="A brief, catchy title for the meeting.")
    summary: str = Field(description="A high-level 3-4 sentence paragraph summarizing the meeting.")
    key_decisions: List[str] = Field(description="A list of the core decisions made during the meeting.")
    action_items: List[ActionItem] = Field(description="A list of actionable tasks assigned to individuals.")


# ==========================================
# BACKEND SPEECH ENGINE
# ==========================================
def transcribe_audio_file(uploaded_file) -> str:
    """Saves the web upload buffer to disk and transcribes it locally."""
    temp_filename = "temp_meeting_recording.mp3"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(temp_filename, beam_size=5)

    full_transcript = []
    for segment in segments:
        full_transcript.append(segment.text)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return " ".join(full_transcript)


# ==========================================
# AI COGNITION ENGINE
# ==========================================
def analyze_transcript(transcript_text: str) -> MeetingSummary:
    """Sends transcript text to Gemini and returns a structured Pydantic object."""
    # Guard check: Ensure key actually exists before hitting the wire
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ API Error: GEMINI_API_KEY is completely missing from your environment setup. Check your .env file.")
        st.stop()

    prompt = (
        "Analyze the following meeting transcript. "
        "Extract the high-level summary, key decisions, and explicit action items with assignees.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            # We add a explicit fallback configuration schema structure compatible across all SDK instances
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MeetingSummary,
                    max_output_tokens=8192,
                    thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
                    temperature=0.2 # Added for output structure consistency
                ),
            )
            if response.parsed is None:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown"
                raise ValueError(f"Gemini returned no structured output (finish_reason={finish_reason}).")

            return response.parsed

        except Exception as e:
            # If Google explicitly tells us why it failed, print the exact text statement to the screen
            error_message = str(e)

            # If it's a transient server issue, retry
            if "503" in error_message and attempt < max_retries - 1:
                st.warning(f"⚠️ API busy. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Provide deep contextual error messages to the UI console for swift debugging
                st.error(f"🔴 Google API Validation Failed: {error_message}")
                if "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
                    st.info("💡 Hint: Go check your `.env` file. Make sure the API key string doesn't contain spaces, quotation marks, or typos.")
                raise e

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="AI Meeting Assistant", page_icon="📋", layout="wide")

st.title("📋 Smart AI Meeting Assistant")
st.caption("Flexible Input (Audio Upload OR Text Paste) ➡️ Gemini AI Summary")

# Create a clean input toggle switcher
input_method = st.radio(
    "💡 Choose your input type:",
    ["🎵 Upload Audio File", "📝 Paste Text Transcript Direct"],
    horizontal=True
)

raw_text = ""  # This variable will hold our final text content regardless of the method chosen

# Render fields interchangeably based on the active selection
if input_method == "🎵 Upload Audio File":
    audio_file = st.file_uploader(label="Upload an audio file (.mp3, .wav, or .m4a):", type=["mp3", "wav", "m4a"])
    text_content_ready = audio_file is not None
else:
    text_area_input = st.text_area(
        label="Paste your meeting text snippet or verbatim dialogue below:",
        height=200,
        placeholder="Alice: We need to ship the code by tomorrow morning.\nBob: I will review the open pull requests now..."
    )
    raw_text = text_area_input
    text_content_ready = len(text_area_input.strip()) > 0

# UNIFIED BUTTON TRIGGER
if st.button("🚀 Process & Extract Insights", type="primary"):
    if not text_content_ready:
        st.warning("Please provide input (either upload an audio file or paste text transcript) before running.")
    else:
        with st.status("🎬 Running Data Pipeline...", expanded=True) as status_box:

            # Conditionally execute transcription only if audio file layout was selected
            if input_method == "🎵 Upload Audio File":
                st.write("🔄 Step 1/2: Processing audio and converting speech to text...")
                raw_text = transcribe_audio_file(audio_file)
            else:
                st.write("✅ Step 1/2 Skipped: Using user provided text transcript directly.")

            # Double check we have extractable text before hitting the API
            if not raw_text.strip():
                status_box.update(label="Pipeline Failed", state="error")
                st.error("The speech engine or input text box contains no processable words.")
            else:
                # Step 2: Route the final text content straight to Gemini
                st.write("🔄 Step 2/2: Structuring transcript with Gemini AI...")
                ai_data: MeetingSummary = analyze_transcript(raw_text)

                status_box.update(label="Pipeline Complete!", state="complete")
                st.balloons()

                # ----------------------------------
                # DYNAMIC RENDER METRICS BELOW
                # ----------------------------------
                st.markdown("---")
                st.header(f"✨ AI Generated Insights: {ai_data.title}")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("💡 Executive Summary & Decisions")
                    st.write(ai_data.summary)

                    st.markdown("#### 🎯 Key Decisions")
                    for decision in ai_data.key_decisions:
                        st.write(f"• {decision}")

                with col2:
                    st.subheader("⚡ Action Items Matrix")
                    table_rows = "".join([
                        f"<tr>"
                        f"<td style='padding:8px; border:1px solid #ddd;'><b>{item.assignee}</b></td>"
                        f"<td style='padding:8px; border:1px solid #ddd;'>{item.task}</td>"
                        f"<td style='padding:8px; border:1px solid #ddd; color:gray;'>{item.deadline}</td>"
                        f"</tr>" for item in ai_data.action_items
                    ])
                    st.markdown(f"""
                    <table style="width:100%; border-collapse: collapse;">
                        <tr style="background-color: #f3f4f6; text-align: left;">
                            <th style="padding:8px; border:1px solid #ddd;">Who</th>
                            <th style="padding:8px; border:1px solid #ddd;">What</th>
                            <th style="padding:8px; border:1px solid #ddd;">Deadline</th>
                        </tr>
                        {table_rows}
                    </table>
                    """, unsafe_allow_html=True)

                # Full Note/Transcript view container block
                st.markdown("---")
                with st.expander("📄 View Full Raw Text Transcript Note", expanded=False):
                    st.text_area(
                        label="Final processed string submitted to AI text node:",
                        value=raw_text,
                        height=250,
                        disabled=True
                    )
