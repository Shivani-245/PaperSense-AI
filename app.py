import time
import os
import re
import html
import hashlib
from datetime import datetime

import streamlit as st
import fitz
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from supabase import create_client

from database import (
    init_db,
    save_paper,
    save_analysis,
    save_question,
    get_recent_papers,
    get_paper_details,
    get_question_history
)


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="PaperSense AI",
    page_icon="📚",
    layout="wide"
)

load_dotenv(".env")

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Restore the logged-in user's session on each Streamlit rerun.
if (
    st.session_state.get("access_token")
    and st.session_state.get("refresh_token")
):
    try:
        supabase.auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"]
        )
    except Exception:
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        st.session_state.pop("user_id", None)

# -------------------- LOGIN / SIGNUP --------------------

if not st.session_state.get("user_id"):

    st.title("📚 PaperSense AI")
    st.subheader("Login or Create Your Account")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")

        if login_submit:
            try:
                result = supabase.auth.sign_in_with_password({
                    "email": email.strip(),
                    "password": password
                })

                if result.session and result.user:
                    st.session_state["access_token"] = result.session.access_token
                    st.session_state["refresh_token"] = result.session.refresh_token
                    st.session_state["user_id"] = result.user.id
                    st.session_state["paper_file_hash"] = None
                    st.session_state["paper_id"] = None
                    st.rerun()
                else:
                    st.error("Login failed. Please try again.")

            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        with st.form("signup_form"):
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input(
                "Create password",
                type="password",
                key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password"
            )
            signup_submit = st.form_submit_button("Create Account")

        if signup_submit:
            if not new_email.strip() or not new_password:
                st.warning("Please enter email and password.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must contain at least 6 characters.")
            else:
                try:
                    result = supabase.auth.sign_up({
                        "email": new_email.strip(),
                        "password": new_password
                    })

                    if result.session and result.user:
                        st.session_state["access_token"] = result.session.access_token
                        st.session_state["refresh_token"] = result.session.refresh_token
                        st.session_state["user_id"] = result.user.id
                        st.rerun()
                    else:
                        st.success(
                            "Account created. Please verify your email, "
                            "then return here and log in."
                        )

                except Exception as e:
                    st.error(f"Signup failed: {e}")

    st.stop()

# Create a user-specific ID for database queries.
user_id = st.session_state["user_id"]

with st.sidebar:
    st.write("Logged in successfully")
    if st.button("Logout"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        for key in [
            "access_token",
            "refresh_token",
            "user_id",
            "paper_id",
            "paper_file_hash"
        ]:
            st.session_state.pop(key, None)

        st.rerun()

init_db()


# -------------------- DATABASE HISTORY --------------------

st.markdown("## 🗂️ Research Paper Database History")

papers = get_recent_papers(supabase, user_id)

if papers:
    st.success(f"Total Research Papers Saved: {len(papers)}")

    paper_options = {
        f"{row[1]} | {row[3]}": row[0]
        for row in papers
    }

    selected_paper = st.selectbox(
        "Select a Previously Uploaded Research Paper",
        options=list(paper_options.keys()),
        key="database_history_select"
    )

  
    selected_id = paper_options[selected_paper]
    details = get_paper_details(supabase, selected_id, user_id)
    

    if details:
        st.markdown("### 📄 Selected Paper Details")

        col1, col2 = st.columns(2)

        col1.metric("Total Pages", details[2])
        col2.write(f"**File Name:** {details[1]}")
        st.write(f"**Uploaded At:** {details[4]}")

        st.markdown("### 📑 Basic Summary")
        st.write(details[5] or "Not available")

        st.markdown("### 🤖 AI Summary")
        st.write(details[6] or "Not generated yet")

        st.markdown("### 📝 Abstract")
        st.write(details[7] or "Not available")

        st.markdown("### 🔑 Keywords")
        st.write(details[8] or "Not available")

        with st.expander("View Extracted PDF Text"):
            st.text_area(
                "Stored Research Paper Content",
                details[3] or "",
                height=300,
                key=f"history_text_{selected_id}"
            )

        st.markdown("### 💬 Question & Answer History")

        history = get_question_history(supabase, selected_id)

        if history:
            for question, answer, created_at in history:
                with st.expander(f"Question: {question}"):
                    st.write(answer or "Answer not available")
                    st.caption(f"Date: {created_at}")
        else:
            st.info("No questions saved for this paper.")

else:
    st.info("No research papers have been saved yet.")

st.markdown("---")


# -------------------- THEME --------------------
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#080d1c,#15152e);color:#f4f4ff}
.block-container{max-width:1150px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3,h4,p,label,li{color:#f4f4ff}
.hero-badge{display:inline-block;padding:8px 16px;border-radius:25px;background:linear-gradient(110deg,#35245f,#203d70);border:1px solid #8065e8;color:#e2d9ff;font-size:13px;font-weight:600;margin-bottom:18px}
.hero-title{font-size:clamp(34px,5vw,52px);font-weight:800;line-height:1.15;color:#fff;margin-bottom:12px}
.hero-description{color:#c4c7df;font-size:17px;margin-bottom:22px}
.keyword-container{display:flex;flex-wrap:wrap;gap:10px;margin:15px 0 25px}
.keyword-tag{display:inline-block;padding:9px 15px;border-radius:25px;background:linear-gradient(110deg,#35245f,#203d70);border:1px solid #8065e8;color:#f0eaff!important;font-size:14px;font-weight:600}
div[data-testid="stMetric"]{background:rgba(42,48,85,.72);border:1px solid #51418b;padding:18px;border-radius:16px}
div[data-testid="stMetricLabel"],div[data-testid="stMetricValue"]{color:#fff!important}
.stButton>button,.stDownloadButton>button{background:linear-gradient(110deg,#713ee8,#2864dc);color:#fff!important;border:1px solid #8065e8;border-radius:10px;padding:9px 20px;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{border:1px solid #c3b5ff;color:#fff!important}
div[data-testid="stFileUploader"]{background:rgba(42,48,85,.55);border:1px solid #8065e8;border-radius:15px;padding:15px}
div[data-testid="stTextInput"] input,div[data-testid="stTextArea"] textarea{background:#f7f8fc!important;color:#172033!important;border:1px solid #9b8bd8!important;border-radius:8px!important}
div[data-testid="stTextInput"] label,div[data-testid="stTextArea"] label{color:#f4f4ff!important}
div[data-testid="stExpander"]{background:rgba(42,48,85,.45);border:1px solid #51418b;border-radius:12px}
div[data-testid="stExpander"] summary,div[data-testid="stExpander"] summary p{color:#fff!important}
</style>
""", unsafe_allow_html=True)

    # -------------------- HELPER FUNCTIONS --------------------
def clean_pdf_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()



def extract_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.getvalue(), filetype="pdf") as doc:
            pages = len(doc)
            text = "\n".join(page.get_text("text") for page in doc)
        return clean_pdf_text(text), pages, None
    except Exception as exc:
        return "", 0, str(exc)


def get_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 8]


def basic_summary(text, limit=5):
    sentences = get_sentences(text)
    if len(sentences) < 2:
        return "Not enough readable sentences were found to create a summary."
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
        chosen = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit])
        return " ".join(sentences[i] for i in chosen)
    except ValueError:
        return "Unable to create a summary from this text."


def extract_keywords(text, limit=15):
    if len(text.split()) < 10:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=limit)
        matrix = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        return sorted([(term, float(score)) for term, score in zip(terms, scores)], key=lambda x: x[1], reverse=True)
    except ValueError:
        return []


def extract_abstract(text):
    pattern = (r"(?ims)^\s*abstract\s*:?[ \t]*(.*?)"
               r"(?=^\s*(?:keywords?|index terms|(?:\d+(?:\.\d+)*)?\s*introduction)\b)")
    match = re.search(pattern, text)
    if match and len(match.group(1).split()) >= 20:
        return match.group(1).strip()[:5000]
    return ""

import time

def gemini_response(prompt):
    if client is None:
        return None, "Gemini API key is not configured. Add GEMINI_API_KEY to Streamlit secrets."

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                return answer.strip(), None

            return None, "Gemini returned an empty response."

        except Exception as exc:
            error_message = str(exc)

            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue

                return None, "Gemini server is currently busy. Please try again after a few minutes."

            return None, error_message

    return None, "Gemini request failed. Please try again."

def similarity_score(text1, text2):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=40000
        )
        matrix = vectorizer.fit_transform([text1, text2])

        return float(
            cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        )

    except ValueError:
        return 0.0


# -------------------- HOMEPAGE --------------------
st.markdown("""
<div class="hero-badge">✦ AI POWERED RESEARCH PLATFORM</div>
<div class="hero-title">Research Smarter.<br>Discover More.</div>
<p class="hero-description">Your intelligent research companion to analyze papers, extract insights and explore scientific knowledge faster.</p>
<p style="color:#d9d2ff">✦ AI Summarization &nbsp; ✦ Smart PDF Analysis &nbsp; ✦ Research Insights</p>

""", unsafe_allow_html=True)
st.markdown("## 📂 Upload Your Research Paper")
st.caption("Upload a PDF document to unlock research analysis.")
uploaded_file = st.file_uploader("Choose your research paper (PDF)", type=["pdf"], key="main_pdf")

if uploaded_file is None:
    st.info("👆 Upload a research paper PDF to get started.")
    st.stop()

st.success(f"PDF uploaded successfully: {uploaded_file.name}")
clean_text, total_pages, pdf_error = extract_pdf(uploaded_file)
if pdf_error:
    st.error(f"PDF Processing Error: {pdf_error}")
    st.stop()
if not clean_text:
    st.warning("No readable text was found. This PDF may be scanned or image-based. OCR is not included in this version.")
    st.stop()
    # Save PDF in database only once per uploaded file
file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

if st.session_state.get("paper_file_hash") != file_hash:

    paper_id = save_paper(
    supabase=supabase,
    user_id=user_id,
    file_name=uploaded_file.name,
    total_pages=total_pages,
    extracted_text=clean_text
)

    st.session_state["paper_file_hash"] = file_hash
    st.session_state["paper_id"] = paper_id

else:
    paper_id = st.session_state["paper_id"]

word_count = len(clean_text.split())
character_count = len(clean_text)
sentence_count = len(re.findall(r"[.!?]+", clean_text))

# -------------------- DOCUMENT OVERVIEW --------------------
st.markdown("## 📊 Document Overview")
st.caption("A quick snapshot of your uploaded research paper.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pages", total_pages)
c2.metric("Total Words", word_count)
c3.metric("Characters", character_count)
c4.metric("Sentences", sentence_count)

# -------------------- BASIC SUMMARY --------------------
st.markdown("## 📄 PDF Summary")
summary = basic_summary(clean_text)
st.write(summary)
st.download_button("📥 Download Summary", summary, "PaperSense_Basic_Summary.txt", "text/plain")

# -------------------- KEYWORDS --------------------
st.markdown("## ✨ Important Keywords")
keywords = extract_keywords(clean_text)
if keywords:
    tags = "".join(f'<span class="keyword-tag">{html.escape(word)}</span>' for word, score in keywords)
    st.markdown(f'<div class="keyword-container">{tags}</div>', unsafe_allow_html=True)
else:
    st.warning("Not enough useful text to extract keywords.")

# -------------------- ABSTRACT --------------------
st.markdown("## 📝 Research Paper Abstract")
abstract = extract_abstract(clean_text)
if abstract:
    st.write(abstract)
else:
    st.info("An Abstract section could not be identified in this PDF.")
    # Save analysis results in database
keyword_names = [word for word, score in keywords]

existing_details = get_paper_details(supabase, paper_id, user_id)
previous_ai_summary = existing_details[6] if existing_details else ""

save_analysis(
    supabase=supabase,
    paper_id=paper_id,
    basic_summary=summary,
    ai_summary=previous_ai_summary,
    abstract=abstract,
    keywords=keyword_names
)

# -------------------- GEMINI SUMMARY --------------------

st.markdown("## 🤖 AI-Powered Summary")

if st.button("Generate AI Summary", key="ai_summary"):
    prompt = f"""Analyze only the supplied research paper. Do not invent facts. If information is missing, say so.
Give simple-English sections: 1. Main Objective 2. Methodology 3. Key Findings 4. Conclusion.
Paper text:\n{clean_text[:18000]}"""

    with st.spinner("Gemini is analyzing your paper..."):
        answer, error = gemini_response(prompt)

    if error:
        st.error(f"Gemini API Error: {error}")
    else:
        st.markdown(answer)

        st.download_button(
            "📥 Download AI Summary",
            answer,
            "PaperSense_AI_Summary.txt",
            "text/plain",
            key="download_ai_summary"
        )

        save_analysis(
            supabase=supabase,
            paper_id=paper_id,
            basic_summary=summary,
            ai_summary=answer,
            abstract=abstract,
            keywords=keyword_names
        )

        st.success("AI Summary saved successfully!")


# -------------------- RESEARCH GAP --------------------

st.markdown("## 🔍 Research Gap Analysis")

if st.button("Analyze Research Gaps", key="research_gap"):

    if client is None:
        st.error("Gemini API key is not configured. Please check your .env file.")

    else:
        prompt = f"""
        Analyze only the supplied research paper.

        Identify:
        1. Main Research Problem
        2. Existing Work Limitations
        3. Research Gaps
        4. Possible Improvements
        5. Future Scope

        Explain in simple English.
        Do not invent information. If something is missing, mention it.

        Research Paper:
        {clean_text[:18000]}
        """

        with st.spinner("Analyzing research gaps..."):
            answer, error = gemini_response(prompt)

        if error:
            st.error(f"Research Gap Analysis Error: {error}")
        else:
            st.markdown("### Research Gap Results")
            st.markdown(answer)

            st.download_button(
                "📥 Download Research Gap Report",
                answer,
                "PaperSense_Research_Gap_Report.txt",
                "text/plain",
                key="download_gap_report"
            )
# -------------------- PDF QUESTION ANSWERING --------------------
st.markdown("## 💬 Ask Questions from PDF")
st.caption("Gemini answers from the uploaded paper when configured; otherwise TF-IDF retrieves relevant passages.")
question = st.text_input("Enter your question about this research paper:", key="pdf_question")
if st.button("Find Answer", key="find_answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
    elif client:
        prompt = f"""Answer the question using only the supplied paper. If the answer is absent, say the paper does not provide enough information. Be concise.
Question: {question}\nPaper text:\n{clean_text[:18000]}"""
        with st.spinner("Finding an answer..."):
            answer, error = gemini_response(prompt)
        if error:
            st.error(f"Question Answering Error: {error}")
        else:
            st.markdown("### 📌 Answer")
            st.write(answer)
        save_question(
    supabase,
    paper_id,
    question.strip(),
    answer
)
    else:
        chunks = [clean_text[i:i + 1800] for i in range(0, len(clean_text), 1400)]
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(chunks + [question])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            indices = scores.argsort()[-3:][::-1]
            if len(indices) == 0 or scores[indices[0]] <= 0:
                st.warning("No relevant information was found in the PDF.")
            else:
                st.markdown("### 📌 Most Relevant Passages")
                for i in sorted(indices):
                    if scores[i] > 0:
                        st.write(chunks[i])
        except ValueError:
            st.warning("Could not process this question. Try different wording.")

# -------------------- PLAGIARISM / SIMILARITY --------------------
st.markdown("## 🧾 Plagiarism & Similarity Check")
st.write("Upload reference PDFs to compare them with the current paper.")
st.caption("This compares only with the reference PDFs you upload. It does not search the internet or a global plagiarism database.")
reference_files = st.file_uploader("Upload reference PDF files", type=["pdf"], accept_multiple_files=True, key="plagiarism_files")
if st.button("Check Similarity", key="check_similarity"):
    if not reference_files:
        st.warning("Upload at least one reference PDF.")
    else:
        refs = []
        for ref_file in reference_files:
            ref_text, ref_pages, ref_error = extract_pdf(ref_file)
            if ref_error:
                st.warning(f"Could not read {ref_file.name}: {ref_error}")
            elif ref_text:
                refs.append((ref_file.name, ref_text))
        if not refs:
            st.warning("No readable text was found in the reference PDFs.")
        else:
            st.markdown("### Similarity Results")
            for name, ref_text in refs:
                score = similarity_score(clean_text, ref_text) * 100
                st.write(f"**{name}** — Text similarity: **{score:.2f}%**")
                st.progress(min(int(round(score)), 100))
            st.info("Similarity is not a final plagiarism verdict. Common technical terms, references, and standard phrases may increase the score. Review matching passages manually.")

# -------------------- MULTIPLE PAPER COMPARISON --------------------
st.markdown("## 📚 Multiple Research Paper Comparison")
st.write("Upload other papers to compare their objectives, methods, findings, and limitations with the current paper.")
comparison_files = st.file_uploader("Upload other research papers (PDF)", type=["pdf"], accept_multiple_files=True, key="comparison_files")
if st.button("Compare Research Papers", key="compare_papers"):
    if not comparison_files:
        st.warning("Upload at least one additional PDF.")
    else:
        documents = [(uploaded_file.name, clean_text)]
        for file in comparison_files:
            text, pages, error = extract_pdf(file)
            if error:
                st.warning(f"Could not read {file.name}: {error}")
            elif text:
                documents.append((file.name, text))
        if len(documents) < 2:
            st.warning("At least two readable papers are required.")
        else:
            st.markdown("### Paper Overview")
            for idx, (name, text) in enumerate(documents, start=1):
                with st.expander(f"Paper {idx}: {name}"):
                    st.write(f"Word count: {len(text.split())}")
                    paper_terms = extract_keywords(text, 8)
                    st.write("Important keywords: " + (", ".join(k for k, _ in paper_terms) if paper_terms else "Not available"))
            st.markdown("### Text Similarity Between Papers")
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    score = similarity_score(documents[i][1], documents[j][1]) * 100
                    st.write(f"**{documents[i][0]}** ↔ **{documents[j][0]}**: {score:.2f}% similarity")
            if client:
                 context = "\n\n".join(f"PAPER {i+1}: {name}\n{text[:6000]}" for i, (name, text) in enumerate(documents))
                 prompt = f"""Compare these research papers using only the supplied text. Do not invent missing details; say 'Not clearly stated' where needed.
Compare: Research Objective, Methodology, Dataset or Study Material, Key Findings, Limitations, Similarities, Differences, Overall Comparison.
{context}"""
                 with st.spinner("Gemini is comparing the papers..."):
                    answer, error = gemini_response(prompt)
                 if error:
                    st.error(f"AI Comparison Error: {error}")
                 else:
                    st.markdown("### AI-Based Comparison")
                    st.markdown(answer)
                    st.download_button("📥 Download Comparison Report", answer, "PaperSense_Paper_Comparison.txt", "text/plain", key="download_comparison")
            else:
                st.info("Gemini is not configured. Basic word counts, keywords, and TF-IDF similarity are shown above.")
                # -------------------- EXTRACTED TEXT --------------------
with st.expander("📖 View Extracted and Cleaned PDF Text"):
    st.text_area("Cleaned Content", clean_text, height=400, key="extracted_content")

# -------------------- DOWNLOAD REPORT --------------------
st.markdown("## 📥 Download Analysis Report")
saved_details = get_paper_details(supabase, paper_id, user_id)

saved_ai_summary = (
    saved_details[6]
    if saved_details and saved_details[6]
    else "AI Summary not generated"
)

qa_records = get_question_history(supabase,paper_id)

qa_report = "\n\n".join(
    [
        f"Question: {q}\nAnswer: {a}\nDate: {date}"
        for q, a, date in qa_records
    ]
) or "No questions asked."
report = f"""PAPERSENSE AI - RESEARCH PAPER ANALYSIS
Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}
Document: {uploaded_file.name}

DOCUMENT OVERVIEW
Total Pages: {total_pages}
Total Words: {word_count}
Characters: {character_count}
Sentences: {sentence_count}

BASIC PDF SUMMARY
{summary}

IMPORTANT KEYWORDS
{', '.join(k for k, _ in keywords) if keywords else 'Not available'}

RESEARCH PAPER ABSTRACT
{abstract if abstract else 'Abstract not identified'}
AI POWERED SUMMARY

{saved_ai_summary}


QUESTION AND ANSWER HISTORY

{qa_report}
"""
st.download_button("Download Complete Basic Report", report, "PaperSense_Complete_Report.txt", "text/plain", key="download_complete_report")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#c4c7df;font-size:13px'>PaperSense AI · Your Intelligent Research Assistant</p>", unsafe_allow_html=True)

