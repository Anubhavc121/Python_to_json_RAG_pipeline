PDF TO JSON AND RAG QUESTION ANSWERING TOOL

OVERVIEW

This project converts one or more PDF files into a structured JSON format containing clean text and meaningful images. It automatically removes repeated logos, decorative graphics, and junk images. On top of this extracted content, it enables Retrieval Augmented Question Answering using local Ollama language models.

The tool is designed mainly for educational PDFs such as textbooks, class notes, coaching material, and slide decks, where correctness and traceability of answers are important.

WHAT THE TOOL DOES

PDF TO STRUCTURED JSON

For every PDF file, the tool extracts the following:

I. Full page text
II. Individual text spans with bounding box coordinates
III. Content-relevant images only

Repeating logos, decorative elements, and very small or noisy images are filtered out automatically. The final output is a clean, structured JSON file for each PDF.

SMART IMAGE PROCESSING

Images extracted from PDFs are processed as follows:

I. Deduplicated using perceptual hashing
II. Classified as content, logo, decoration, or junk
III. Logos and decorations removed by default
IV. Optional base64 embedding of images inside JSON
V. Optional image captions generated using a vision model such as llava

OCR FALLBACK SUPPORT

If a page contains very little extractable text, for example scanned pages or image-only slides, the tool applies OCR using Tesseract. The OCR text is merged into the page text automatically so that it becomes searchable and usable for question answering.

RETRIEVAL AUGMENTED QUESTION ANSWERING

All extracted text is processed as follows:

I. Text is split into manageable chunks
II. Embeddings are created using SentenceTransformers
III. Relevant chunks are retrieved for each question
IV. A local Ollama language model generates the answer
V. Answers are strictly grounded in the PDF context

SUPPORTED QUESTION TYPES

The system supports the following question types:

I. True or False
II. Assertion–Reason
III. Multiple Choice Questions with options a to d
IV. Open-ended descriptive questions

For exam-style questions, strict formatting rules are enforced to avoid ambiguous answers.

SOURCE ATTRIBUTION

When enabled, every answer includes:

I. Source PDF name
II. Page or slide number
III. Internal chunk identifier

This makes the tool suitable for students, teachers, and exam preparation workflows where answer traceability is required.

KEY FEATURES

Key features of the system include:

I. Recursive directory processing of PDFs
II. Multi-PDF indexing
III. Automatic removal of logos and decorations
IV. OCR fallback for scanned content
V. Strict exam-style answer formatting
VI. Fully local execution using Ollama with no cloud APIs
VII. Interactive question-answering loop

EXAMPLE USE CASE

A typical use case includes:

I. Converting Class 6 mathematics PDFs into structured JSON
II. Asking exam-style questions such as True or False, Assertion–Reason, or MCQs
III. Receiving answers with exact page or slide references

INSTALLATION

Install the required Python dependencies using pip:

pip install pymupdf pillow numpy sentence-transformers pydantic tqdm ollama pytesseract

OCR SETUP

For OCR support, install Tesseract:

I. On macOS, run: brew install tesseract
II. On Ubuntu or Debian, run: sudo apt install tesseract-ocr

OLLAMA SETUP

Install Ollama and pull the required models:

ollama pull llama3
ollama pull llava:7b

USAGE

To process a directory of PDFs and start interactive question answering, run:

python Pdf_json.py --input_dir /path/to/pdfs --out output_dir --loop --ocr --show_sources

To process a single PDF and save the extracted JSON, run:

python Pdf_json.py --pdf notes.pdf --out output_dir --save_json

To ask a single question without entering interactive mode, run:

python Pdf_json.py --input_dir ./pdfs --ask "True or False: A square has equal sides."

OUTPUT STRUCTURE

The output directory is organized as follows:

I. One folder per PDF
II. Each folder contains an images directory with extracted content images
III. Each folder contains a JSON file with the full structured extraction

DESIGN PRINCIPLES

The system follows these design principles:

I. Local-first execution with no external APIs
II. Deterministic and reproducible JSON output
III. Robust handling of noisy or scanned PDFs
IV. Education-focused correctness checks
V. Scalable to large PDF collections
