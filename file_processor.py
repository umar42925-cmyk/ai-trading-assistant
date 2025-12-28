import base64
import io
from pathlib import Path

def process_file(file_data: dict) -> dict:
    """
    Process uploaded file and extract content.
    Returns: {
        "type": "image" | "text" | "pdf" | "unknown",
        "content": str or base64,
        "filename": str,
        "mime_type": str
    }
    """
    filename = file_data.get("name", "unknown")
    mime_type = file_data.get("type", "")
    data = file_data.get("data")
    
    if not data:
        return {"type": "unknown", "content": "", "filename": filename, "mime_type": mime_type}
    
    # Image files
    if mime_type.startswith("image/"):
        return {
            "type": "image",
            "content": base64.b64encode(data).decode(),
            "filename": filename,
            "mime_type": mime_type
        }
    
    # PDF files
    if mime_type == "application/pdf":
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return {
                "type": "pdf",
                "content": text.strip(),
                "filename": filename,
                "mime_type": mime_type
            }
        except:
            return {
                "type": "pdf",
                "content": f"[PDF file: {filename} - content extraction failed]",
                "filename": filename,
                "mime_type": mime_type
            }
    
    # Text-based files
    text_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.css', '.xml', '.yaml', '.yml', '.sh', '.bat'}
    file_ext = Path(filename).suffix.lower()
    
    if mime_type.startswith("text/") or file_ext in text_extensions:
        try:
            text = data.decode('utf-8')
            return {
                "type": "text",
                "content": text,
                "filename": filename,
                "mime_type": mime_type
            }
        except:
            return {
                "type": "text",
                "content": f"[Text file: {filename} - encoding error]",
                "filename": filename,
                "mime_type": mime_type
            }
    
    # Word documents
    if mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            text = "\n".join([para.text for para in doc.paragraphs])
            return {
                "type": "text",
                "content": text,
                "filename": filename,
                "mime_type": mime_type
            }
        except:
            return {
                "type": "unknown",
                "content": f"[Word document: {filename} - extraction failed]",
                "filename": filename,
                "mime_type": mime_type
            }
    
    # Excel files
    if mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(data))
            text = df.to_string()
            return {
                "type": "text",
                "content": f"Excel data:\n{text}",
                "filename": filename,
                "mime_type": mime_type
            }
        except:
            return {
                "type": "unknown",
                "content": f"[Excel file: {filename} - extraction failed]",
                "filename": filename,
                "mime_type": mime_type
            }
    
    # Unknown file type
    return {
        "type": "unknown",
        "content": f"[File: {filename} ({mime_type}) - unsupported format]",
        "filename": filename,
        "mime_type": mime_type
    }


def format_file_for_llm(processed_file: dict, user_message: str = "") -> str:
    """
    Format processed file content for LLM input.
    """
    file_type = processed_file.get("type")
    filename = processed_file.get("filename")
    content = processed_file.get("content")
    
    if file_type == "image":
        return f"{user_message}\n\n[User uploaded image: {filename}]\nNote: Image processing not yet implemented."
    
    elif file_type in ["text", "pdf"]:
        return f"{user_message}\n\n--- File Content: {filename} ---\n{content}\n--- End of File ---"
    
    else:
        return f"{user_message}\n\n{content}"