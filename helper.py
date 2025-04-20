import os
import google.generativeai as genai
import PyPDF2 as pdf
import docx
import textract
import json

def configure_genai(api_key):
    """Configure the Generative AI API with error handling."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to configure Generative AI: {str(e)}")
    

def get_gemini_response(prompt):
    """Generate a response using Gemini with enhanced error handling and response validation."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        config = genai.types.GenerationConfig(
            temperature=0
        )
        response = model.generate_content(prompt, generation_config=config)
        
        # Ensure response is not empty
        if not response or not response.text:
            raise Exception("Empty response received from Gemini")
            
        # Try to parse the response as JSON
        try:
            response_json = json.loads(response.text)
            
            # Validate required fields
            required_fields = ["JD Match", "MissingKeywords","MatchingKeywords", "Profile Summary"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field: {field}")
                    
            return response.text
            
        except json.JSONDecodeError:
            # If response is not valid JSON, try to extract JSON-like content
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response.text, re.DOTALL)
            if match:
                return match.group()
            else:
                raise Exception("Could not extract valid JSON response")
                
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or DOC file."""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "pdf":
            reader = pdf.PdfReader(uploaded_file)
            if len(reader.pages) == 0:
                raise Exception("PDF file is empty")
            text = [page.extract_text() for page in reader.pages if page.extract_text()]
            if not text:
                raise Exception("No text could be extracted from the PDF")
            return " ".join(text)

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = [para.text for para in doc.paragraphs if para.text.strip()]
            if not text:
                raise Exception("No text found in DOCX file")
            return "\n".join(text)

        elif file_type == "doc":
            # Save to temp location
            with open("temp.doc", "wb") as f:
                f.write(uploaded_file.read())
            text = textract.process("temp.doc").decode("utf-8")
            os.remove("temp.doc")
            if not text.strip():
                raise Exception("No text found in DOC file")
            return text

        else:
            raise Exception("Unsupported file type. Only PDF, DOC, and DOCX are allowed.")

    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")


def prepare_prompt(resume_text, job_description):
    """Prepare the input prompt with improved structure and validation."""
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")
        
    prompt_template = """
    Act as an expert ATS (Applicant Tracking System) and resume analysis engine.

    Evaluate the following resume against the provided job description.

    Analyze and extract:
    - A match score percentage (`JD Match`)
    - Missing keywords that are in the JD but not in the resume (`MissingKeywords`)
    - Matching keywords that appear in both (`MatchingKeywords`)
    - A full and detailed profile summary (`Profile Summary`), which includes:
        - Strengths
        - Weaknesses
        - Suggestions for improvement
        - Resume structure quality
        - Clarity and impact of writing
        - Relevance to job requirements
    - An explanation of the JD match score and how it was calculated ('ScoreExplanation')

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Respond ONLY in this JSON format:
    {{
        "JD Match": "percentage between 0-100",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "MatchingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "Detailed and thorough evaluation covering strengths, weaknesses, and improvement suggestions.",
        "ScoreExplanation": "Provide a clear explanation of how the score was calculated. Include a breakdown of the score based on different components such as:\n
            - Keyword Match (e.g., matched 10/14 keywords = 71%)\n
            - Technical Skills Match (e.g., 4/6 required tools/technologies found = 67%)\n
            - Resume Structure & Presentation (e.g., clear formatting, quantified achievements = 80%)\n
            - Overall Relevance to Job Role (e.g., experience aligns with job = 75%)\n
            Add details about what aspects improved the score and what reduced it. Mention missing skills, weak structure, or other factors that influenced the final percentage."

    }}
    """
    
    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )