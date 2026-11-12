import os
import io
import json
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & ATS Score Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .score-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #cbd5e1;
        margin-bottom: 20px;
    }
    .skill-badge-match {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 3px;
        border: 1px solid #c8e6c9;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 3px;
        border: 1px solid #fecaca;
    }
    .suggestion-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: API Key Configuration & Instructions
with st.sidebar:
    st.header("⚙️ Configuration")

    env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    api_key = ""

    if env_api_key:
        st.success("Gemini API Key detected from `.env`", icon="✅")
        # Allow optional override if desired
        override_key = st.text_input(
            "Override API Key (optional):",
            type="password",
            placeholder="Leave empty to use .env key",
            help="If you wish to temporarily use a different key, enter it here.",
        )
        api_key = override_key.strip() if override_key.strip() else env_api_key
    else:
        st.warning("No API Key detected in `.env`", icon="⚠️")
        user_input_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            placeholder="AIzaSy...",
            help="Your API key will remain in session memory and is never hardcoded.",
        )
        api_key = user_input_key.strip()
        st.caption(
            "💡 **Tip:** Create a `.env` file containing `GEMINI_API_KEY=your_key` to avoid manual entry."
        )

    st.markdown("---")
    st.subheader("📌 How to Use")
    st.markdown(
        """
        1. Ensure your Gemini API Key is loaded or provided above.
        2. Upload candidate resume in **PDF** format.
        3. Paste the target **Job Description (JD)**.
        4. Click **Analyze Resume & Check ATS Score**.
        5. Review ATS match metrics, skills breakdown, and actionable improvements.
        """
    )

    st.markdown("---")
    st.caption("Powered by **Gemini 2.5 Flash** & Streamlit")


# Header Section
st.markdown('<div class="main-title">📄 AI Resume Analyzer & ATS Score Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Optimize candidate resumes against target job descriptions with intelligent ATS scoring, keyword gap analysis, and tailored recommendations.</div>',
    unsafe_allow_html=True,
)


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from uploaded PDF using pypdf PdfReader."""
    try:
        reader = PdfReader(uploaded_file)
        text_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
        return "\n".join(text_pages).strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if returned by the model."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def analyze_with_gemini(api_key: str, resume_text: str, job_description: str) -> dict:
    """Calls Gemini 2.5 Flash via google-genai SDK to analyze resume against JD."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert Applicant Tracking System (ATS) evaluator and corporate recruiting specialist.
Analyze the following candidate resume text against the target job description.

--- RESUME TEXT ---
{resume_text}

--- TARGET JOB DESCRIPTION ---
{job_description}

--- INSTRUCTIONS ---
Perform a strict ATS evaluation and return your response EXCLUSIVELY in valid JSON format matching this exact schema:
{{
  "ats_score": <int between 0 and 100>,
  "score_summary": "<brief 1-2 sentence explanation of the score>",
  "matching_skills": [
    "<skill or keyword found in both the resume and the job description>"
  ],
  "missing_skills": [
    "<important keyword or skill required/preferred in the job description that is missing or underrepresented in the resume>"
  ],
  "actionable_suggestions": [
    "<Actionable suggestion 1 to optimize the resume specifically for this role>",
    "<Actionable suggestion 2 to optimize the resume specifically for this role>",
    "<Actionable suggestion 3 to optimize the resume specifically for this role>",
    "<Actionable suggestion 4 to optimize the resume specifically for this role>",
    "<Actionable suggestion 5 to optimize the resume specifically for this role>"
  ]
}}

Requirements:
1. "ats_score" must be an integer from 0 to 100 based on keyword overlap, experience alignment, and qualification fit.
2. "matching_skills" must list recognized skills/technologies/tools present in both.
3. "missing_skills" must highlight important requirements from the JD absent in the resume.
4. "actionable_suggestions" must contain EXACTLY 5 clear, high-impact, specific suggestions.
5. Output ONLY the JSON object. Do not include introductory text or markdown formatting.
"""

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=config,
        )
    except Exception:
        # Fallback if config parameters vary
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )

    response_text = response.text if hasattr(response, "text") else str(response)
    cleaned = clean_json_text(response_text)
    return json.loads(cleaned)


# Main Layout: 2 Columns for Inputs
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF format)",
        type=["pdf"],
        help="Upload the candidate's resume in PDF format to extract its text content.",
    )

    resume_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:
            st.success(
                f"✅ Extracted {len(resume_text.split())} words across {len(PdfReader(uploaded_file).pages)} page(s)."
            )
            with st.expander("👁️ View Extracted Resume Text"):
                st.text_area(
                    "Extracted Text Preview",
                    value=resume_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )
        else:
            st.error(
                "Could not extract text from this PDF. Please ensure it is not a scanned image without selectable text."
            )

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area(
        "Paste the Job Description (JD):",
        height=260,
        placeholder="Paste the complete job description here, including responsibilities, requirements, and preferred qualifications...",
        help="Paste the full job posting to ensure accurate ATS keyword matching and gap analysis.",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Analyze Button
analyze_button = st.button("🚀 Analyze Resume & Check ATS Score", type="primary", use_container_width=True)

if analyze_button:
    # Validation checks
    if not api_key:
        st.error(
            "🔑 **Gemini API Key missing!** Please enter your API key in the sidebar or define `GEMINI_API_KEY` in your `.env` file."
        )
    elif not uploaded_file:
        st.warning("⚠️ Please upload a PDF resume before proceeding.")
    elif not resume_text:
        st.warning("⚠️ No readable text found in the uploaded resume. Please check the PDF file.")
    elif not job_description.strip():
        st.warning("⚠️ Please paste the target Job Description before proceeding.")
    else:
        with st.spinner("Analyzing resume against job description using Gemini 2.5 Flash..."):
            try:
                results = analyze_with_gemini(api_key, resume_text, job_description.strip())
                st.session_state["ats_results"] = results
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

# Display Results if available in session state
if "ats_results" in st.session_state:
    results = st.session_state["ats_results"]
    ats_score = int(results.get("ats_score", 0))
    score_summary = results.get("score_summary", "")
    matching_skills = results.get("matching_skills", [])
    missing_skills = results.get("missing_skills", [])
    actionable_suggestions = results.get("actionable_suggestions", [])

    st.markdown("---")
    st.header("📊 ATS Evaluation Results")

    # 1. ATS Match Score Section
    st.subheader("1. ATS Match Score")
    score_col1, score_col2 = st.columns([1, 2], gap="medium")

    with score_col1:
        st.metric(
            label="Overall ATS Match Score",
            value=f"{ats_score} / 100",
            delta=f"{ats_score}% Match",
        )

    with score_col2:
        # Determine score status and color
        if ats_score >= 80:
            badge_color = "🟢"
            status_text = "Strong Match — Highly aligned with JD requirements."
        elif ats_score >= 60:
            badge_color = "🟡"
            status_text = "Moderate Match — Consider closing keyword and skill gaps."
        else:
            badge_color = "🔴"
            status_text = "Low Match — Significant alignment gaps; resume revision recommended."

        st.markdown(f"**Fit Assessment:** {badge_color} *{status_text}*")
        # Display progress bar
        normalized_progress = min(max(ats_score / 100.0, 0.0), 1.0)
        st.progress(normalized_progress)

        if score_summary:
            st.caption(f"📝 {score_summary}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 & 3. Matching and Missing Skills
    skills_col1, skills_col2 = st.columns(2, gap="large")

    with skills_col1:
        st.subheader(f"2. Matching Skills Found ({len(matching_skills)})")
        st.markdown(
            "Skills and qualifications present in **both the resume and the job description**:"
        )
        if matching_skills:
            for skill in matching_skills:
                st.markdown(f"- ✅ **{skill}**")
        else:
            st.info("No direct skill matches were identified.")

    with skills_col2:
        st.subheader(f"3. Missing Skills / Keywords ({len(missing_skills)})")
        st.markdown(
            "Important keywords from the job description **missing or underrepresented** in the resume:"
        )
        if missing_skills:
            for skill in missing_skills:
                st.markdown(f"- ⚠️ **{skill}**")
        else:
            st.success("Great job! No major required skills appear to be missing.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. 5 Actionable Suggestions
    st.subheader("4. 5 Actionable Suggestions to Improve Your Resume")
    st.markdown(
        "Exactly 5 clear, specific recommendations to optimize your resume for this role:"
    )

    if actionable_suggestions:
        suggestion_icons = ["🎯", "✍️", "💡", "📌", "⚡"]
        for idx, suggestion in enumerate(actionable_suggestions[:5], start=1):
            icon = suggestion_icons[(idx - 1) % len(suggestion_icons)]
            st.markdown(
                f"""
                <div class="suggestion-box">
                    <strong>{icon} Suggestion {idx}:</strong> {suggestion}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No suggestions generated.")
