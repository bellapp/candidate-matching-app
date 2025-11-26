#!/usr/bin/env python3
"""
Streamlit Application for Candidate Matching Score

This app allows users to:
1. Input a job posting
2. Upload a candidate's CV (PDF)
3. View extracted rubric criteria
4. See scores for each criterion
5. View the final matching score
"""

import streamlit as st
import sys
import os
from pathlib import Path
import io
import json

# Add the current directory to path to import from test_matching_score
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from test_matching_score
# Note: We'll need to dynamically update the API key in the module
import test_matching_score
from test_matching_score import (
    extract_rubric_with_llm,
    score_criteria_with_llm,
    calculate_matching_score,
    generate_qualification_note,
    EvaluationRubric,
    CriterionScore
)

# PDF extraction - try both libraries
PDF_LIBRARY = None
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    pass

try:
    import pdfplumber
    if PDF_LIBRARY is None:
        PDF_LIBRARY = "pdfplumber"
except ImportError:
    pass

if PDF_LIBRARY is None:
    # Will show error when user tries to upload
    pass

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        st.session_state['env_loaded'] = True
except ImportError:
    pass

# Initialize Langfuse for observability
langfuse = None
LANGFUSE_ENABLED = False

try:
    from langfuse import Langfuse
   # from langfuse.decorators import observe
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if public_key and secret_key:
        try:
            langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            LANGFUSE_ENABLED = True
            st.sidebar.success("✓ Langfuse observability enabled (v3.x)")
            st.sidebar.caption(f"Host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Langfuse initialization failed: {str(e)}")
            langfuse = None
            LANGFUSE_ENABLED = False
    else:
        st.sidebar.info("ℹ️ Langfuse not configured (optional)")
except ImportError:
    langfuse = None
    LANGFUSE_ENABLED = False
    st.sidebar.info("ℹ️ Langfuse not installed (optional)")

# Set API key from environment or session state before importing
# This needs to be done before importing test_matching_score
if "OPENROUTER_API_KEY" not in os.environ:
    # Will be set from Streamlit input later
    os.environ["OPENROUTER_API_KEY"] = ""



def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    if PDF_LIBRARY is None:
        st.error("⚠️ Please install a PDF library: `pip install PyPDF2` or `pip install pdfplumber`")
        return ""
    
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        if PDF_LIBRARY == "PyPDF2":
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        else:  # pdfplumber
            # Create a copy of file bytes for pdfplumber
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return ""


def format_score_color(score: int) -> str:
    """Return color based on score."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "orange"
    else:
        return "red"


def main():
    st.set_page_config(
        page_title="Candidate Matching Score",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Candidate Matching Score")
    st.markdown("Evaluate candidate CVs against job postings using AI-powered criteria scoring")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
       
        api_key = os.getenv("OPENROUTER_API_KEY")
        test_matching_score.OPENROUTER_API_KEY = api_key
        
        st.divider()
        
        # Model selection
        st.subheader("🤖 AI Model")
        from test_matching_score import (
            CLAUDE_HAIKU_OPENROUTER,
            GEMINI_FLASH_OPENROUTER,
            GEMINI_FLASH_LITE_OPENROUTER,
            GPT_OSS_120B_OPENROUTER,
            MODEL_NAMES
        )
        
        model_options = {
            MODEL_NAMES[GPT_OSS_120B_OPENROUTER]: GPT_OSS_120B_OPENROUTER,
            MODEL_NAMES[GEMINI_FLASH_LITE_OPENROUTER]: GEMINI_FLASH_LITE_OPENROUTER,
            MODEL_NAMES[CLAUDE_HAIKU_OPENROUTER]: CLAUDE_HAIKU_OPENROUTER,
            MODEL_NAMES[GEMINI_FLASH_OPENROUTER]: GEMINI_FLASH_OPENROUTER,
            
            
        }
        
        selected_model_name = st.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            help="Different models have different speeds, costs, and capabilities"
        )
        selected_model = model_options[selected_model_name]
        
        st.caption(f"Selected: `{selected_model}`")
        
        st.divider()
        
        # Cache option
        use_cache = st.checkbox("Use Cache", value=True, help="Cache rubric extraction for faster repeated evaluations")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Delete all cached rubrics"):
            import shutil
            cache_dir = Path(__file__).parent / ".rubric_cache"
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    st.success("✅ Cache cleared successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to clear cache: {e}")
            else:
                st.info("ℹ️ Cache is already empty")
        
        # Prompt version selector (if Langfuse is enabled)
        prompt_version = None
        prompt_label = None
        if LANGFUSE_ENABLED:
            st.subheader("🔬 Prompt Version")
            version_mode = st.radio(
                "Select prompt version:",
                options=["Latest (default)", "Version 1", "Version 2", "Custom Label"],
                help="Compare different prompt versions from Langfuse"
            )
            
            if version_mode == "Version 1":
                prompt_version = 1
            elif version_mode == "Version 2":
                prompt_version = 2
            elif version_mode == "Custom Label":
                prompt_label = st.text_input("Label:", value="production", help="e.g., 'production', 'latest', 'experiment'")
        
        st.divider()
        st.markdown("### 📊 About")
        # st.markdown("""
        # This tool uses AI to:
        # 1. Extract evaluation criteria from job postings
        # 2. Score candidates against each criterion
        # 3. Calculate a weighted matching score
        # """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Job Posting")
        job_posting = st.text_area(
            "Enter the job posting text:",
            height=300,
            placeholder="Paste the full job description here..."
        )
    
    with col2:
        st.header("📄 Candidate CV")
        
        # Input method tabs
        input_method = st.radio(
            "Choose input method:",
            options=["📎 Upload PDF", "📝 Paste Text"],
            horizontal=True,
            help="Upload a PDF file or paste CV text directly"
        )
        
        cv_text = None
        uploaded_file = None
        
        if input_method == "📎 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload CV (PDF)",
                type=["pdf"],
                help="Upload the candidate's CV as a PDF file"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                
                # Extract text from PDF
                with st.spinner("Extracting text from PDF..."):
                    cv_text = extract_text_from_pdf(uploaded_file)
                
                if cv_text:
                    with st.expander("📄 Preview extracted text"):
                        st.text_area(
                            "Extracted CV Text:",
                            value=cv_text,
                            height=200,
                            disabled=True
                        )
                    st.info(f"✅ Extracted {len(cv_text)} chars, ~{len(cv_text.split())} words")
                else:
                    st.error("Failed to extract text from PDF")
        else:
            # Paste text option
            cv_text = st.text_area(
                "Paste CV text here:",
                height=300,
                placeholder="Paste the candidate's complete CV text here...\n\nExample:\nJohn Doe\nSenior Software Engineer\n\nExperience:\n- Company A (2020-2023): Led team of 5...\n- Company B (2018-2020): Developed...",
                help="Paste the complete CV text directly"
            )
            
            if cv_text:
                st.info(f"✅ CV text received ({len(cv_text)} chars, ~{len(cv_text.split())} words)")
    
    # Process button
    st.divider()
    
    if st.button("🚀 Evaluate Candidate", type="primary", width="stretch"):
        # Validation
        if not api_key:
            st.error("❌ Please enter your OpenRouter API Key in the sidebar")
            st.stop()
        
        if not job_posting.strip():
            st.error("❌ Please enter a job posting")
            st.stop()
        
        if not cv_text or not cv_text.strip():
            st.error("❌ Please upload a CV PDF file")
            st.stop()
        
        # Process
        progress_bar = st.progress(0)
        status_text = st.empty()
        timing_container = st.empty()
        
        # Track timing for each step
        import time
        step_times = {}
        total_start = time.time()
        
        try:
            candidate_name = uploaded_file.name if uploaded_file else "unknown"
            
            # LANGFUSE: Generate session ID for grouping all observations
            session_id = None
            if LANGFUSE_ENABLED and langfuse is not None:
                import hashlib
                # Create unique session ID for this evaluation
                session_id = f"streamlit-{hashlib.md5(f'{candidate_name}-{time.time()}'.encode()).hexdigest()[:12]}"
                st.info(f"🔍 Langfuse tracking | Session: `{session_id}` | Version: {prompt_version or 'latest'}")
            
            # Step 1: Extract Rubric
            step1_start = time.time()
            status_text.text("📋 Step 1/4: Extracting evaluation criteria from job posting...")
            progress_bar.progress(20)
            
            with st.spinner("Analyzing job posting..."):
                rubric = extract_rubric_with_llm(
                    job_posting,
                    use_cache=use_cache,
                    prompt_version=prompt_version,
                    prompt_label=prompt_label,
                    session_id=session_id,
                    model=selected_model
                )
            
            step_times['rubric_extraction'] = time.time() - step1_start
            timing_container.info(f"⏱️ Step 1 completed in {step_times['rubric_extraction']:.2f}s")
            
            progress_bar.progress(40)
            
            # Step 2: Score Criteria
            step2_start = time.time()
            status_text.text("📊 Step 2/4: Scoring candidate against criteria...")
            progress_bar.progress(60)
            
            with st.spinner("Evaluating candidate..."):
                criteria_scores = score_criteria_with_llm(
                    cv_text, 
                    rubric,
                    session_id=session_id,
                    model=selected_model
                )
            
            step_times['criteria_scoring'] = time.time() - step2_start
            timing_container.info(f"⏱️ Steps 1-2 completed in {sum(step_times.values()):.2f}s (Step 2: {step_times['criteria_scoring']:.2f}s)")
            
            progress_bar.progress(70)
            
            # Step 3: Calculate Final Score
            step3_start = time.time()
            status_text.text("🎯 Step 3/4: Calculating final matching score...")
            progress_bar.progress(80)
            
            result = calculate_matching_score(rubric, criteria_scores)
            
            step_times['score_calculation'] = time.time() - step3_start
            timing_container.info(f"⏱️ Steps 1-3 completed in {sum(step_times.values()):.2f}s (Step 3: {step_times['score_calculation']:.2f}s)")
            
            progress_bar.progress(85)
            
            # Step 4: Generate Qualification Note
            step4_start = time.time()
            status_text.text("📝 Step 4/4: Generating qualification note...")
            progress_bar.progress(90)
            
            with st.spinner("Generating comprehensive qualification assessment..."):
                qualification_note = generate_qualification_note(
                    job_posting,
                    cv_text,
                    session_id=session_id,
                    model=selected_model
                )
            
            step_times['qualification_generation'] = time.time() - step4_start
            total_time = time.time() - total_start
            
            # Display final timing summary
            timing_container.success(f"""
            ⏱️ **Total Time: {total_time:.2f}s**
            - Step 1 (Rubric Extraction): {step_times['rubric_extraction']:.2f}s
            - Step 2 (Criteria Scoring): {step_times['criteria_scoring']:.2f}s
            - Step 3 (Score Calculation): {step_times['score_calculation']:.2f}s
            - Step 4 (Qualification Note): {step_times['qualification_generation']:.2f}s
            """)
            
            # LANGFUSE: Flush any pending events
            if LANGFUSE_ENABLED and langfuse is not None:
                try:
                    langfuse.flush()
                    st.success(f"✅ Evaluation complete! Session: {session_id}")
                    st.info("💡 View in Langfuse: https://cloud.langfuse.com → Sessions")
                except Exception as e:
                    pass
            
            progress_bar.progress(100)
            status_text.text("✅ Evaluation complete!")
            
            # Display Results
            st.divider()
            st.header("📊 Results")
            
            # Final Score - Large Display
            final_score = result["final_score"]
            score_color = format_score_color(final_score)
            
            col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
            with col_score2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="color: {score_color}; margin-bottom: 10px;">Final Matching Score</h2>
                    <h1 style="font-size: 72px; color: {score_color}; margin: 0;">{final_score}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Rubric Criteria
            st.subheader("📋 Evaluation Criteria")
            st.markdown(f"**Total Criteria:** {len(rubric.criteria)} | **Total Weight:** {rubric.total_weight:.1f}%")
            
            criteria_df_data = []
            for criterion in rubric.criteria:
                criteria_df_data.append({
                    "Criterion": criterion.name,
                    "Weight": f"{criterion.weight:.1f}%",
                    "Description": criterion.description,
                    "Required": "✅ Yes" if criterion.is_required else "⚪ Preferred"
                })
            
            st.dataframe(
                criteria_df_data,
                width="stretch",
                hide_index=True
            )
            
            st.divider()
            
            # Summary Table
            st.subheader("📊 Score Summary")
            summary_data = []
            empty_evidence_count = 0
            
            for breakdown_item in result["breakdown"]:
                # Get evidence and gap, ensuring they're strings
                evidence = breakdown_item.get("evidence", "") or ""
                gap = breakdown_item.get("gap", "") or ""
                
                # Track empty evidence
                if not evidence:
                    empty_evidence_count += 1
                
                # Use fallback text if empty
                if not evidence:
                    evidence = "No evidence provided by AI"
                if not gap and breakdown_item["score"] < 80:
                    gap = "Score below 80 - review recommended"
                
                summary_data.append({
                    "Criterion": breakdown_item["criterion"],
                    "Score": breakdown_item["score"],
                    "Weight": f"{breakdown_item['weight']:.1f}%",
                    "Contribution": f"{breakdown_item['contribution']:.2f}",
                    "Evidence": evidence,
                    "Gap": gap
                })
            
            # Show warning if many empty evidence fields
            if empty_evidence_count > 0:
                st.warning(f"⚠️ {empty_evidence_count} out of {len(summary_data)} criteria have no evidence. The AI may not be returning evidence/gap fields correctly.")
            
            # Show debug info if needed
            with st.expander("🔍 Debug Info (Click to see raw data)"):
                st.json({
                    "breakdown_sample": result["breakdown"][0] if result["breakdown"] else None,
                    "criteria_scores_sample": {
                        "criteria_name": criteria_scores[0].criteria_name if criteria_scores else None,
                        "score": criteria_scores[0].score if criteria_scores else None,
                        "evidence": criteria_scores[0].evidence if criteria_scores else None,
                        "gap": criteria_scores[0].gap if criteria_scores else None,
                    } if criteria_scores else None,
                    "all_criteria_scores": [
                        {
                            "criteria_name": cs.criteria_name,
                            "score": cs.score,
                            "evidence": cs.evidence,
                            "gap": cs.gap
                        }
                        for cs in criteria_scores
                    ]
                })
            
            st.dataframe(
                summary_data,
                width="stretch",
                hide_index=True
            )
            
            # Download results as JSON
            st.divider()
            
            # Qualification Note Display
            st.header("📝 Candidate Qualification Note")
            st.markdown("""
            This comprehensive assessment provides a detailed analysis of the candidate's fit for the role,
            based on recent experience, career trajectory, and requirements alignment.
            """)
            
            # Clean and display the HTML-formatted qualification note
            import re
            
            cleaned_note = qualification_note.strip()
            
            # Remove code block markers if present
            if "```html" in cleaned_note:
                cleaned_note = cleaned_note.split("```html")[1].split("```")[0].strip()
            elif "```" in cleaned_note:
                cleaned_note = cleaned_note.split("```")[1].split("```")[0].strip()
            
            # Remove any <div> wrappers from the LLM response (including the qual-note-container div)
            cleaned_note = re.sub(r'<div[^>]*>', '', cleaned_note)
            cleaned_note = cleaned_note.replace('</div>', '')
            
            # Remove any remaining HTML artifacts that might show as raw text
            cleaned_note = re.sub(r'^\s*<[^>]+>\s*$', '', cleaned_note, flags=re.MULTILINE)
            
            print(f"✓ Cleaned note (first 300 chars): {cleaned_note[:300]}")
            
            # Add CSS styling for better presentation
            qualification_html = f"""
            <style>
                .qual-note-container {{
                    background-color: #f8f9fa;
                    padding: 25px;
                    border-radius: 10px;
                    border-left: 5px solid #007bff;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .qual-note-container b {{
                    color: #1e3a8a;
                    font-size: 1.15em;
                    display: block;
                    margin: 20px 0 10px 0;
                }}
                .qual-note-container ul {{
                    margin: 10px 0;
                    padding-left: 25px;
                    list-style-type: disc;
                }}
                .qual-note-container li {{
                    margin: 10px 0;
                    line-height: 1.7;
                    color: #374151;
                }}
                .qual-note-container p {{
                    margin: 12px 0;
                    line-height: 1.8;
                    color: #374151;
                }}
                .qual-note-container li b {{
                    display: inline;
                    font-size: 1em;
                    margin: 0;
                }}
            </style>
            
                {cleaned_note}
            
            """
            
            # Display with proper HTML rendering
            st.markdown(qualification_html, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("💾 Export Results")
            
            results_json = {
                "final_score": final_score,
                "qualification_note": qualification_note,
                "rubric": {
                    "criteria": [
                        {
                            "name": c.name,
                            "weight": c.weight,
                            "description": c.description,
                            "is_required": c.is_required
                        }
                        for c in rubric.criteria
                    ]
                },
                "criteria_scores": [
                    {
                        "criteria_name": cs.criteria_name,
                        "score": cs.score,
                        "evidence": cs.evidence,
                        "gap": cs.gap
                    }
                    for cs in criteria_scores
                ],
                "breakdown": result["breakdown"]
            }
            
            st.download_button(
                label="📥 Download Results as JSON",
                data=json.dumps(results_json, indent=2),
                file_name="matching_score_results.json",
                mime="application/json"
            )
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {str(e)}")
            st.error("The AI response could not be parsed as JSON. This might be due to:")
            st.markdown("""
            - The API returned an error message instead of JSON
            - The response format was unexpected
            - Network issues during API call
            """)
            st.exception(e)
            progress_bar.empty()
            status_text.empty()
        except ValueError as e:
            error_msg = str(e)
            st.error(f"❌ Error: {error_msg}")
            if "JSON parsing failed" in error_msg or "Empty response" in error_msg:
                st.warning("💡 **Troubleshooting:**")
                st.markdown("""
                - Check your OpenRouter API key is valid
                - Verify you have API credits available
                - Try again - sometimes the API needs a retry
                - Check the console/logs for the raw API response
                """)
            st.exception(e)
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
            st.exception(e)
            progress_bar.empty()
            status_text.empty()


if __name__ == "__main__":
    main()

