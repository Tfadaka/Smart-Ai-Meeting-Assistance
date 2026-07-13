import os
import time
import streamlit as st
from google import genai
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# 1. Load keys and initialize the modern Google SDK
load_dotenv()
client = genai.Client()

# 2. Define data structure models using Pydantic
class ActionItem(BaseModel):
    task: str = Field(description="The specific task that needs to be done.")
    assignee: str = Field(description="The person responsible for this task. Use 'Unassigned' if unknown.")
    deadline: str = Field(description="The mentioned deadline, or 'Not specified'.")

class MeetingSummary(BaseModel):
    title: str = Field(description="A brief, catchy title for the meeting.")
    summary: str = Field(description="A high-level 3-4 sentence paragraph summarizing the meeting.")
    key_decisions: List[str] = Field(description="A list of the core decisions made during the meeting.")
    action_items: List[ActionItem] = Field(description="A list of actionable tasks assigned to individuals.")

# 3. Audio Transcription Engine (Runs locally for free)
def transcribe_audio_file(uploaded_file) -> str:
    temp_filename = "temp_meeting_recording.mp3"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Using 'tiny' for fast execution inside cloud container CPUs
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(temp_filename, beam_size=5)

    full_transcript = []
    for segment in segments:
        full_transcript.append(segment.text)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return " ".join(full_transcript)

# 4. Intelligence Engine (Connects safely to Gemini API with retry safeguards)
def analyze_transcript(transcript_text: str) -> MeetingSummary:
    prompt = (
        "Analyze the following meeting transcript. "
        "Extract the high-level summary, key decisions, and explicit action items with assignees.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": MeetingSummary,
                },
            )
            return response.parsed
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                st.warning(f"⚠️ API busy. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise e

# 5. User Interface Presentation Layer
st.set_page_config(page_title="AI Meeting Assistant", page_icon="📋", layout="wide")
st.title("📋 Smart AI Meeting Assistant")
st.caption("Audio Upload ➡️ Local Transcription ➡️ Gemini Summary Pipeline")

st.subheader("🎵 Upload Meeting Recording")
audio_file = st.file_uploader(label="Upload an audio file (.mp3, .wav, or .m4a):", type=["mp3", "wav", "m4a"])

# 6. Single-Button Chained Pipeline Trigger
if st.button("🚀 Process Recording & Extract Minutes", type="primary"):
    if audio_file is None:
        st.warning("Please upload a valid audio recording file first.")
    else:
        with st.status("🎬 Running Pipeline...", expanded=True) as status_box:

            st.write("🔄 Step 1/2: Processing audio and converting to text...")
            raw_text = transcribe_audio_file(audio_file)

            if not raw_text.strip():
                status_box.update(label="Pipeline Failed", state="error")
                st.error("The speech engine could not extract any words from the file.")
            else:
                st.write("🔄 Step 2/2: Passing text to Gemini API for structured summary...")
                ai_data: MeetingSummary = analyze_transcript(raw_text)

                status_box.update(label="Pipeline Complete!", state="complete")
                st.balloons()

                # Render results below the button
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

                # Collapsible container for full text note view
                st.markdown("---")
                with st.expander("📄 View Full Raw Text Transcript Note", expanded=False):
                    st.text_area(label="Verbatim text extracted from audio:", value=raw_text, height=250, disabled=True)
