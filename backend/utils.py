import aiofiles
import urllib
import mistune
import os

async def write_to_file(filename: str, text: str) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # 确保输入内容是字符串。
    if not isinstance(text, str):
        text = str(text)

    # 转换为 UTF-8，并替换无法编码的字符。
    text_utf8 = text.encode('utf-8', errors='replace').decode('utf-8')

    async with aiofiles.open(filename, "w", encoding='utf-8') as file:
        await file.write(text_utf8)

async def write_text_to_md(text: str, filename: str = "") -> str:
    """Writes text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    import uuid

    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.md"
    await write_to_file(file_path, text)
    return urllib.parse.quote(file_path)

def _preprocess_images_for_pdf(text: str) -> str:
    """Convert web image URLs to absolute file paths for PDF generation.
    
    Transforms /outputs/images/... URLs to absolute file:// paths that
    weasyprint can resolve.
    """
    import re
    
    base_path = os.path.abspath(".")
    
    # 匹配使用 /outputs/ 路径的 Markdown 图片。
    def replace_image_url(match):
        """把一个站内输出图片 URL 改写为 PDF 渲染器可读的绝对路径。"""
        alt_text = match.group(1)
        url = match.group(2)
        
        # 将 /outputs/... 转换为绝对路径。
        if url.startswith("/outputs/"):
            abs_path = os.path.join(base_path, url.lstrip("/"))
            return f"![{alt_text}]({abs_path})"
        return match.group(0)
    
    # 匹配 ![alt text](/outputs/images/...) 格式。
    pattern = r'!\[([^\]]*)\]\((/outputs/[^)]+)\)'
    return re.sub(pattern, replace_image_url, text)


async def write_md_to_pdf(text: str, filename: str = "") -> str:
    """Converts Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated PDF.
    """
    import uuid

    # 空文件名会生成 outputs/.pdf，容易导致下载界面混淆；使用稳定的非空文件名。
    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    # 替换路径分隔符，确保 PDF 保存在 outputs/ 目录下。
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.pdf"

    try:
        # 相对于后端模块解析 CSS 路径，避免依赖当前工作目录。
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "styles", "pdf_styles.css")
        
        # 预处理图片 URL，使其兼容 PDF 渲染器。
        processed_text = _preprocess_images_for_pdf(text)
        
        # 将当前目录设为基准路径，以解析剩余的相对路径。
        base_url = os.path.abspath(".")
        from md2pdf.core import md2pdf
        md2pdf(
               file_path,
               raw=processed_text,
               css=css_path,
               base_url=base_url,
            )
        print(f"Report written to {file_path}")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return ""

    encoded_file_path = urllib.parse.quote(file_path)
    return encoded_file_path

async def write_md_to_word(text: str, filename: str = "") -> str:
    """Converts Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated DOCX.
    """
    import uuid

    safe_name = (filename or "").strip()[:60] or f"report-{uuid.uuid4().hex[:12]}"
    safe_name = safe_name.replace("/", "-").replace("\\", "-")
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{safe_name}.docx"

    try:
        from docx import Document
        from htmldocx import HtmlToDocx
        # 将报告 Markdown 转换为 HTML。
        html = mistune.html(text)
        # 创建 DOCX 文档对象。
        doc = Document()
        # 将报告生成的 HTML 转换为文档格式。
        HtmlToDocx().add_html_to_document(html, doc)

        # 将 DOCX 文档保存到目标路径。
        doc.save(file_path)

        print(f"Report written to {file_path}")

        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path

    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e}")
        return ""
