from nicegui import ui, app
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
import json
from datetime import datetime
import time

# Configure app
app.add_static_files('/static', 'static')

# Global variables
model = None
doc_embeddings = None
chunks = None
model_loaded = False

class DiagramEngine:
    def __init__(self):
        self.model = None
        self.doc_embeddings = None
        self.chunks = None        
    def load_document(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as doc:
                return doc.read()
        except Exception as e:
            return None
    
    def chunk_by_sections(self, text):
        sections = text.split("\n\n")
        sections = [s.strip() for s in sections if len(s.strip()) > 0]
        return sections
    
    def initialize_model(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def embed_document(self, document_path):
        try:
            document = self.load_document(document_path)
            if document is None:
                return False
            
            self.chunks = self.chunk_by_sections(document)
            self.doc_embeddings = self.model.encode(
                self.chunks, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            return True
        except Exception as e:
            print(f"Error embedding document: {e}")
            return False
    
    def retrieve_top(self, query, threshold=0.2):
        try:
            query_embeddings = self.model.encode(
                [query], 
                normalize_embeddings=True, 
                convert_to_numpy=True
            )[0]
            similarities = np.dot(self.doc_embeddings, query_embeddings)
            best_index = np.argmax(similarities)
            
            if similarities[best_index] < threshold:
                return None, similarities[best_index]
            
            return self.chunks[best_index], similarities[best_index]
        except Exception as e:
            print(f"Error retrieving: {e}")
            return None, 0
    
    def construct_prompt(self, retrieved, query):
        return f"""You are a STRICT Mermaid flowchart code generator.

Context reference:
{retrieved if retrieved else "No context available"}

User request:
{query}

OUTPUT RULES:
- Output ONLY Mermaid code.
- Do NOT include explanations.
- Do NOT include Markdown formatting.
- Do NOT include code fences.
- Start directly with "flowchart" or "graph".
- Generate a valid flowchart diagram.
- Do not describe the diagram.
- Do not add narrative text.
- If unsure, generate a simple Start → End flowchart.

Example valid output:
flowchart TD
    Start([Start]) --> Process[Process]
    Process --> End([End])
"""
    
    def generate_with_ollama(self, prompt):
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Generation Error: {e}")
            return None

# Initialize engine
engine = DiagramEngine()

# ============ STYLING ============
ui.query('body').style('margin: 0; padding: 0; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;')

# Custom CSS for animations and theme
custom_css = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
    min-height: 100vh;
}

.hero-section {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 50%, rgba(36, 36, 62, 0.95) 100%);
}

.hero-content {
    text-align: center;
    z-index: 10;
    animation: slideUp 0.8s ease-out;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #a0aec0;
    margin-bottom: 2rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

.cta-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2.5rem;
    font-size: 1.1rem;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
    animation: popIn 0.6s ease-out 0.2s both;
}

.cta-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.cta-button:hover::before {
    left: 100%;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
}

.cta-button:active {
    transform: translateY(0);
}

.app-section {
    display: none;
    animation: fadeIn 0.5s ease-out forwards;
}

.app-section.active {
    display: block;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    animation: slideInUp 0.6s ease-out;
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(102, 126, 234, 0.3);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
}

.input-group {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.input-group input {
    flex: 1;
    min-width: 250px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 12px;
    padding: 1rem;
    color: #e0e0e0;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.input-group input:focus {
    outline: none;
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(102, 126, 234, 0.6);
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
}

.input-group input::placeholder {
    color: #718096;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.btn-primary:active {
    transform: translateY(0);
}

.btn-secondary {
    background: transparent;
    color: #667eea;
    border: 2px solid #667eea;
    padding: 0.8rem 1.8rem;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: rgba(102, 126, 234, 0.1);
    transform: translateY(-2px);
}

dagram-container {
    background: rgba(255, 255, 255, 0.03);
    border: 2px solid rgba(102, 126, 234, 0.2);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
    min-height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: popIn 0.6s ease-out;
}

.mermaid {
    display: flex;
    justify-content: center;
    align-items: center;
}

.loading-spinner {
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

.status-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 1rem;
    animation: popIn 0.4s ease-out;
}

.status-success {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-info {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.result-metadata {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    flex-wrap: wrap;
}

.metadata-item {
    display: flex;
    flex-direction: column;
}

.metadata-label {
    font-size: 0.9rem;
    color: #718096;
    margin-bottom: 0.25rem;
}

.metadata-value {
    font-size: 1.1rem;
    color: #667eea;
    font-weight: 600;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin-top: 3rem;
}

.feature-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: popIn 0.6s ease-out;
}

.feature-card:nth-child(1) { animation-delay: 0.1s; }
.feature-card:nth-child(2) { animation-delay: 0.2s; }
.feature-card:nth-child(3) { animation-delay: 0.3s; }
.feature-card:nth-child(4) { animation-delay: 0.4s; }

.feature-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(102, 126, 234, 0.3);
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #667eea;
}

.feature-desc {
    color: #a0aec0;
    font-size: 0.95rem;
    line-height: 1.5;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes popIn {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.footer {
    text-align: center;
    padding: 2rem;
    color: #718096;
    margin-top: 4rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

glowing-text {
    color: #667eea;
    text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
    }
    
    .container {
        padding: 1rem;
    }
    
    .glass-card {
        padding: 1.5rem;
    }
    
    .input-group {
        flex-direction: column;
    }
    
    .features-grid {
        grid-template-columns: 1fr;
    }
}
"""

# Inject CSS
ui.add_head_html(f'<style>{custom_css}</style>')
ui.add_head_html('<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>')
ui.add_head_html("""
<script>
mermaid.initialize({ startOnLoad: true, theme: 'dark' });
</script>
""")

# ============ LAYOUT ============

@ui.page('/')
def landing_page():
    with ui.element('div').classes('hero-section'):
        with ui.element('div').classes('hero-content'):
            ui.html('<h1 class="hero-title">Query to Diagram</h1>')
            ui.html('''<p class="hero-subtitle">
                Transform your natural language queries into beautiful Mermaid flowchart diagrams.
                Powered by semantic search and AI-driven code generation.
            </p>''')
            
            # Features
            with ui.element('div').classes('features-grid'):
                features = [
                    ('🚀', 'Fast & Efficient', 'Get diagrams in seconds'),
                    ('🎨', 'Beautiful Diagrams', 'Modern, clean visual design'),
                    ('🧠', 'Smart Retrieval', 'Semantic search for context'),
                    ('⚡', 'Real-time Generation', 'Instant diagram rendering')
                ]
                
                for icon, title, desc in features:
                    with ui.element('div').classes('feature-card'):
                        ui.html(f'<div class="feature-icon">{icon}</div>')
                        ui.html(f'<div class="feature-title">{title}</div>')
                        ui.html(f'<div class="feature-desc">{desc}</div>')
            
            # CTA Button
            ui.button('Start Creating', on_click=lambda: navigate_to_app()).classes('cta-button')

@ui.page('/app')
def app_page():
    def navigate_to_landing():
        ui.navigate.to('/')
    
    def navigate_to_app():
        ui.navigate.to('/app')
    
    with ui.header().classes('w-full'):
        ui.label('Query to Diagram').style('font-size: 1.5rem; font-weight: bold; color: #667eea;')
        ui.link('← Back to Home', '/').style('color: #667eea;')
    
    with ui.element('div').classes('container'):
        # Load Model Section
        with ui.element('div').classes('glass-card'):
            ui.label('Step 1: Initialize Model').style('font-size: 1.2rem; font-weight: 600; color: #667eea;')
            
            with ui.row().classes('w-full gap-4'):
                doc_path_input = ui.input(label='Document Path', placeholder='Path to your document').classes('flex-grow')
                model_status = ui.label('Not loaded').classes('status-badge status-info')
            
            def load_model():
                path = doc_path_input.value
                if not path:
                    model_status.text = 'Please enter a document path'
                    model_status.classes('status-badge status-error', remove='status-info')
                    return
                
                model_status.text = 'Loading...'
                model_status.classes('status-badge status-info')
                
                if engine.initialize_model():
                    if engine.embed_document(path):
                        model_status.text = '✓ Model Ready'
                        model_status.classes('status-badge status-success', remove='status-info status-error')
                    else:
                        model_status.text = '✗ Document Load Failed'
                        model_status.classes('status-badge status-error', remove='status-info status-success')
                else:
                    model_status.text = '✗ Model Load Failed'
                    model_status.classes('status-badge status-error', remove='status-info status-success')
            
            ui.button('Load Model', on_click=load_model).classes('btn-primary')
        
        # Query Section
        with ui.element('div').classes('glass-card'):
            ui.label('Step 2: Enter Your Query').style('font-size: 1.2rem; font-weight: 600; color: #667eea;')
            
            query_input = ui.textarea(label='Query', placeholder='Describe the process or algorithm you want diagrammed...').classes('w-full')
            query_input.style('min-height: 120px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(102, 126, 234, 0.3); border-radius: 12px; padding: 1rem; color: #e0e0e0;')
            
            def generate_diagram():
                if not query_input.value:
                    ui.notify('Please enter a query')
                    return
                
                if engine.doc_embeddings is None:
                    ui.notify('Please load the model first')
                    return
                
                # Show loading state
                spinner = ui.spinner(size='lg')
                status_label = ui.label('Generating diagram...').classes('status-badge status-info')
                
                # Retrieve context
                retrieved, score = engine.retrieve_top(query_input.value)
                
                # Construct and generate
                prompt = engine.construct_prompt(retrieved, query_input.value)
                diagram_code = engine.generate_with_ollama(prompt)
                
                if diagram_code:
                    status_label.text = f'✓ Generated (Confidence: {score:.2f})'
                    status_label.classes('status-badge status-success', remove='status-info')
                    
                    # Display diagram
                    with ui.element('div').classes('diagram-container'):
                        ui.html(f'''
                        <div class="mermaid">
                        {diagram_code}
                        </div>
                        ''')
                    
                    # Download button
                    def download_diagram():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'diagram_{timestamp}.html'
                        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 2rem;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }}
        .mermaid {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }}
    </style>
</head>
<body>
    <div class="mermaid">
    {diagram_code}
    </div>
</body>
</html>'''  
                        with open(filename, 'w') as f:
                            f.write(html_content)
                        ui.notify(f'Diagram saved as {filename}')
                    
                    ui.button('Download HTML', on_click=download_diagram).classes('btn-primary')
                else:
                    status_label.text = '✗ Generation Failed'
                    status_label.classes('status-badge status-error', remove='status-info')
                    ui.notify('Failed to generate diagram')
            
            ui.button('Generate Diagram', on_click=generate_diagram).classes('btn-primary')
        
        # Footer
        ui.html('''
        <div class="footer">
            <p>Query to Diagram Engine • Powered by Sentence Transformers & Mistral AI</p>
            <p style="font-size: 0.9rem; color: #4a5568;">
                Make sure Ollama is running with the Mistral model: <code style="background: rgba(0,0,0,0.3); padding: 0.25rem 0.5rem; border-radius: 4px;">ollama run mistral</code>
            </p>
        </div>
        ''')

def navigate_to_app():
    ui.navigate.to('/app')

# Run the application
if __name__ == '__main__':
    ui.run(host='0.0.0.0', port=8000, dark=True)