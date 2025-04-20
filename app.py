import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import json
from dotenv import load_dotenv
from helper import configure_genai, get_gemini_response, extract_text_from_file, prepare_prompt

def init_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    init_session_state()
    
    # Configure Generative AI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY in your .env file")
        return
        
    try:
        configure_genai(api_key)
    except Exception as e:
        st.error(f"Failed to configure API: {str(e)}")
        return

    # Sidebar
    with st.sidebar:
        st.title("Resume screening and analyzer using Artificial Intelligence")
        st.subheader("Welcome to resume analyzer")
        st.write("""
        This software helps you to:
        - Evaluate resume-job description match
        - get a resume matching score against the job description and quality
        - Get information about strengths and weakness of the resume and further improvement suggestions
        """)

    # Main content
    st.title("Resume screening and analyzer using Artificial Intelligence")
    st.subheader("Optimize Your Resume for ATS")
    
    # Input sections with validation
    jd = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        help="Enter the complete job description for accurate analysis"
    )
    
    uploaded_file = st.file_uploader(
        "Resume (PDF, DOC, DOCX)",
        type=["pdf", "doc", "docx"],
        help="Upload your resume in PDF, DOC or DOCX format"
    )

    # Process button with loading state
    if st.button("Analyze Resume", disabled=st.session_state.processing):
        if not jd:
            st.warning("Please provide a job description.")
            return
            
        if not uploaded_file:
            st.warning("Please upload a resume in PDF format.")
            return
            
        st.session_state.processing = True
        
        try:
            with st.spinner("📊 Analyzing your resume..."):
                # Extract text from PDF
                resume_text = extract_text_from_file(uploaded_file)
                
                # Prepare prompt
                input_prompt = prepare_prompt(resume_text, jd)
                
                # Get and parse response
                response = get_gemini_response(input_prompt)
                response_json = json.loads(response)
                
                # Display results
                st.success("✨ Analysis Complete!")
                
                # Match percentage
                match_percentage = response_json.get("JD Match", "N/A")
                st.metric("Match Score", match_percentage)
                
                # Missing keywords
                st.subheader("Missing Keywords")
                missing_keywords = response_json.get("MissingKeywords", [])
                if missing_keywords:
                    st.write(", ".join(missing_keywords))
                else:
                    st.write("No critical missing keywords found!")

                # Matching Keywords
                st.subheader("Matching Keywords")
                matching_keywords = response_json.get("MatchingKeywords", [])
                if matching_keywords:
                    st.write(", ".join(matching_keywords))
                else:
                    st.write("No critical matching keywords found!")
                
                # Profile summary
                st.subheader("Profile Summary")
                st.write(response_json.get("Profile Summary", "No summary available"))

                st.subheader("ScoreExplanation")
                st.write(response_json.get("ScoreExplanation", "can't explain the score"))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()