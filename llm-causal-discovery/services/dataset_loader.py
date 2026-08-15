import os
from pypdf import PdfReader

def read_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    try:
        content = []

        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    content.append(f"{text}")

        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content.append(f.read())

        return "\n\n".join(content).strip()

    except Exception as e:
        raise Exception(f"Erro ao ler o arquivo {file_path}: {str(e)}")