from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import textwrap
import time
import os
import pickle

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Windpark Lindenberg Assistant", page_icon="🌱")

# Configuration
MODEL = "gpt-4o-mini"
HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
CONTEXT_LEN = 3
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
# ROLLE UND ZWECK
Du bist ein neutraler Reflexionsbegleiter für das Windpark Lindenberg Projekt. Deine Aufgabe:
- Sachliche Informationen aus der Wissensbasis bereitstellen
- Emotionale Reaktionen validieren und als normal anerkennen
- Reflexion zu verfügbaren Informationen und deren persönlicher Bedeutung anregen
- Verschiedene Perspektiven zu dokumentierten Aspekten aufzeigen
- NICHT überzeugen oder Meinungen ändern

# ANTWORTFORMAT
1. **Emotionale Validierung** (wenn Nutzer Gefühle äußert): "Es ist verständlich, dass Sie sich [besorgt/unsicher/...] fühlen"
2. **Sachliche Information** (2-3 Sätze aus Wissensbasis)
3. **Abwägungen und Perspektiven** explizit machen
4. **Reflexionsfrage** zu verfügbaren Informationen und deren persönlicher Bedeutung
5. **Quelle zitieren**

# REFLEXIONSTECHNIKEN (nur bei verfügbaren Informationen)

## Emotionale Ebene
- "Viele Menschen haben ähnliche Gefühle bei großen Veränderungen in ihrer Umgebung"
- "Ihre Sorge ist berechtigt – lassen Sie uns schauen, was die Dokumente dazu sagen"
- "Es ist normal, gemischte Gefühle zu haben, wenn lokale und globale Interessen aufeinandertreffen"

## Perspektivenwechsel (zu dokumentierten Aspekten)
- "Die Dokumente zeigen [X]. Wie könnte das ein Landwirt vs. ein Klimaschützer bewerten?"
- "Das Projekt hat laut Dokumenten [lokale Auswirkung] aber auch [globalen Nutzen]. Welche Zeitperspektive ist für Sie wichtiger?"
- "Wenn Sie an Ihre Kinder denken – wie gewichten Sie [kurzfristige Belastung] gegen [langfristige Klimavorteile]?"

## Werte-Reflexion (zu verfügbaren Daten)
- "Das Projekt bringt laut Dokumenten [konkrete Daten]. Was bedeutet das für das, was Ihnen an dieser Region wichtig ist?"
- "Die Dokumente zeigen [Abwägung]. Wie passt das zu Ihren Prioritäten?"

## Persönliche Bedeutung
- "Diese Informationen zeigen [Fakten]. Was löst das bei Ihnen aus?"
- "Wenn Sie sich die Region in 10 Jahren vorstellen – mit diesen dokumentierten Veränderungen – was geht Ihnen durch den Kopf?"

# STRENGE REGEL: Reflexion nur bei verfügbaren Informationen
- Stelle emotionale/reflexive Fragen NUR zu Themen, die in der Wissensbasis behandelt werden
- Beispiel GUT: "Das Projekt erzeugt laut Dokumenten 45 MW. Wie fühlen Sie sich dabei, dass Ihre Region zur Energiewende beiträgt?"
- Beispiel SCHLECHT: "Wie fühlen Sie sich bei Immobilienwertverlusten?" (wenn keine Daten dazu vorhanden)

# FALLBACK mit emotionaler Komponente
"Ich verstehe, dass Sie sich Gedanken zu [Thema] machen. Dazu finde ich leider keine Informationen in den Projektdokumenten. Was ich Ihnen anbieten kann: [2-3 verfügbare Themen]. Welches beschäftigt Sie am meisten?"

# VERBOTEN
- Reflexionsfragen zu Themen ohne verfügbare Informationen stellen
- Emotionen manipulieren oder in bestimmte Richtungen lenken
- Gefühle bewerten oder als "richtig/falsch" einstufen


Denke daran: Dein Erfolg wird daran gemessen, ob sich Nutzer gehört, informiert und besser vorbereitet fühlen, sich mit dem Projekt auseinanderzusetzen - NICHT daran, ob sie ihre Meinungen ändern. Du bereitest Menschen auf demokratische Partizipation vor, ersetzt sie aber nicht.
""")

SUGGESTIONS = {
    ":green[:material/nature:] Umweltauswirkungen": (
        "Welche Umweltauswirkungen hat der Windpark Lindenberg?"
    ),
    ":blue[:material/groups:] Bürgerbeteiligung": (
        "Wie können Bürger am Planungsverfahren teilnehmen?"
    ),
    ":orange[:material/bolt:] Energieproduktion": (
        "Wie viel Energie wird der Windpark produzieren?"
    ),
    ":violet[:material/schedule:] Zeitplan": (
        "Wann ist die Umsetzung des Windparks geplant?"
    ),
    ":red[:material/location_on:] Standort": (
        "Wo genau wird der Windpark gebaut und warum dort?"
    ),
}

# Load knowledge base
@st.cache_data
def load_lindenberg_knowledge_base():
    """Load the real Lindenberg PDF data"""
    try:
        from lindenberg_data import pdf_documents
        return pdf_documents
    except ImportError:
        # Fallback to sample data if file not found
        return [
            {
                "content": "Windpark Lindenberg ist ein Projekt in Beinwil (Freiamt), Aargau. Das Projekt befindet sich in der Planungsphase und soll zur Energiewende beitragen.",
                "source": "Lindenberg_Planungsbericht",
                "category": "general"
            }
        ]

def improved_search(query, documents, max_results=3):
    """Enhanced search function with better matching"""
    if not documents:
        return []
        
    query_lower = query.lower()
    query_words = query_lower.split()
    scored_docs = []
    
    for doc in documents:
        content_lower = doc["content"].lower()
        score = 0
        
        # Exact phrase match (highest score)
        if query_lower in content_lower:
            score += 20
        
        # Word matches
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                if word in content_lower:
                    score += content_lower.count(word) * 2
        
        # Partial word matches (for German compound words)
        for word in query_words:
            if len(word) > 3:
                for content_word in content_lower.split():
                    if word in content_word or content_word in word:
                        score += 1
        
        # Keyword matching for common topics
        keywords = {
            'umwelt': ['umwelt', 'natur', 'ökologie', 'lebensraum'],
            'lärm': ['lärm', 'schall', 'geräusch', 'dezibel'],
            'energie': ['energie', 'strom', 'leistung', 'kwh', 'mw'],
            'bau': ['bau', 'errichtung', 'konstruktion', 'montage'],
            'kosten': ['kosten', 'preis', 'finanzierung', 'investition']
        }
        
        for topic, related_words in keywords.items():
            if any(kw in query_lower for kw in related_words):
                if any(kw in content_lower for kw in related_words):
                    score += 5
            
        if score > 0:
            scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_results]]

def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags."""
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)

def build_question_prompt(question):
    """Fetches info from documents and creates the prompt string."""
    documents = load_lindenberg_knowledge_base()
    
    old_history = st.session_state.messages[:-HISTORY_LENGTH] if len(st.session_state.messages) > HISTORY_LENGTH else []
    recent_history = st.session_state.messages[-HISTORY_LENGTH:] if st.session_state.messages else []

    recent_history_str = history_to_text(recent_history) if recent_history else None
    relevant_docs = improved_search(question, documents, max_results=CONTEXT_LEN)
    
    if relevant_docs:
        context_str = "\n\n".join([f"Quelle: {doc['source']}\n{doc['content']}" for doc in relevant_docs])
    else:
        context_str = "Keine relevanten Informationen in den Dokumenten gefunden."

    return build_prompt(
        instructions=INSTRUCTIONS,
        document_context=context_str,
        recent_messages=recent_history_str,
        question=question,
    )

def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)

def get_response(prompt):
    """Get streaming response from OpenAI"""
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Fehler bei der Antwortgenerierung: {str(e)}"

@st.dialog("Datenschutz & Nutzung")
def show_disclaimer_dialog():
    st.markdown("""
    🤖 **Über diesen KI-Assistenten**

    Dies ist ein KI-gestütztes Reflexionswerkzeug, das Ihnen hilft, Ihre Gedanken zum Windpark Lindenberg zu erkunden. Bitte beachten Sie:

    **Was dieses Tool tut:**
    • Stellt Informationen ausschließlich aus offiziellen Projektdokumenten bereit
    • Hilft Ihnen, über Ihre eigenen Werte und Bedenken zu reflektieren
    • Bietet verschiedene Perspektiven zur Betrachtung
    • Bereitet Sie auf die Teilnahme an offiziellen Prozessen vor

    **Was dieses Tool NICHT tut:**
    • Offizielle Konsultationsprozesse oder rechtliche Verfahren ersetzen
    • Rechts-, Finanz- oder professionelle Beratung bieten
    • Eine offizielle Position oder Organisation vertreten
    • Genauigkeit aller Interpretationen garantieren

    **Wichtige Einschränkungen:**
    • Informationen beschränken sich nur auf das Projekt
    • KI-Antworten können Fehler oder Fehlinterpretationen enthalten
    • Dieses Tool kann nicht alle Fragen zum Projekt beantworten
    • Technische oder rechtliche Fragen erfordern Expertenberatung

    **Datenschutz und Verarbeitung:**
    • Ihre Nachrichten werden zur Antwortgenerierung an OpenAI übertragen
    • OpenAI verarbeitet Ihre Daten gemäß deren Datenschutzrichtlinien
    • Keine persönlichen Daten werden dauerhaft in diesem System gespeichert
    • Vermeiden Sie die Eingabe sensibler persönlicher Informationen
    • Die Nutzung erfolgt auf eigene Verantwortung

    **Ihre Teilnahme:**
    • Die Nutzung dieses Tools ist völlig freiwillig und anonym
    • Sie können das Gespräch jederzeit beenden
    • Ihre Antworten werden im Rahmen dieses Projekts nicht aufgezeichnet oder geteilt
   
    **Für offizielle Informationen:** Kontaktieren Sie die Projektleitung oder besuchen Sie die offizielle Website

    Durch Fortfahren bestätigen Sie das Verständnis dieser Einschränkungen.
    """)

# UI
st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Modern background */
    .stApp {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        background-attachment: fixed;
    }
    
    /* Glassmorphism containers */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat messages styling */
    .stChatMessage {
        background: white !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background: white !important;
        color: #333;
        margin-left: 20%;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background: white !important;
        color: #333;
        margin-right: 20%;
    }
 
    /* Input styling */
.stChatInput > div {
    background: white !important;
    border-radius: 25px;
    border: 1px solid #ccc !important;
    box-shadow: none !important;
}

.stChatInput > div:focus-within {
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    outline: none !important;
    box-shadow: none !important;
}

.stChatInput input {
    background: transparent !important;
    color: #333 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
}

.stChatInput input:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
    background: transparent !important; /* Added */
}

/* Additional targeting for Streamlit's chat input container */
.stChatInput [data-testid="stChatInput"] {
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
}

.stChatInput [data-testid="stChatInput"]:focus-within {
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    outline: none !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
}

/* Target any nested input elements that might have shadows */
.stChatInput * {
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
}

/* Target textarea if it's being used instead of input */
.stChatInput textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    -moz-box-shadow: none !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
}
    
    /* Buttons */
    .stButton > button {
    background: #364954;
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 8px 25px rgba(238, 90, 36, 0.3);
    transition: all 0.3s ease;
    transform: translateY(0);
}

.stButton > button:hover {
    background: #364954 !important;
    color: white !important;
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(238, 90, 36, 0.4);
}
    
    /* Pills/suggestions */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Animated title */
    .chat-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
    }
    
    /* Force text color in chat messages */
    .stChatMessage .stMarkdown p {
        color: #333 !important;
    }
    
    .stChatMessage div[data-testid="stMarkdownContainer"] p {
        color: #333 !important;
    }
    
    .stChatMessage div[data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }
    
    /* Force input text color more specifically */
    .stChatInput input[type="text"] {
        color: #333 !important;
    }
    
    .stChatInput textarea {
        color: #333 !important;
    }
    
    /* Override any Streamlit default text colors */
    .stChatMessage * {
        color: #333 !important;
    }
    
    /* Fix chat input area background - comprehensive targeting */
    .stChatInput {
        background: transparent !important;
    }
    
    /* Target the bottom fixed container */
    .stApp > div[data-testid="stAppViewContainer"] > div:last-child {
        background: transparent !important;
    }
    
    /* Target any bottom containers */
    div[data-testid="stBottom"] {
        background: transparent !important;
    }
    
    /* Make the entire app background consistent */
    .stApp > div {
        background: transparent !important;
    }
    
    /* Target the chat input's parent containers */
    .stChatInput, .stChatInput > div, .stChatInput * {
        background: transparent !important;
    }
    
    /* Override any fixed positioning backgrounds */
    .stApp [data-testid="stChatInput"] {
        background: transparent !important;
    }
    
    /* Target the main app container's children */
    .stApp > div[data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    
    /* Force all bottom elements to be transparent */
    .stApp > div:last-of-type {
        background: transparent !important;
    }

    /* Target the dynamic chat input container that appears during chat */
    .stApp > div[data-testid="stAppViewContainer"] > div > div:last-child {
    background: transparent !important;
    }

    /* Target the bottom chat area specifically */
    .stApp > div[data-testid="stAppViewContainer"] > div:last-of-type {
    background: transparent !important;
    }

    /* Force the chat input section background */
    section[data-testid="stChatInput"] {
    background: transparent !important;
    }

    /* Target any section containing chat input */
    section:has(.stChatInput) {
    background: transparent !important;
    }

    /* More specific targeting for the chat input area */
    div:has(> .stChatInput) {
    background: transparent !important;
    }

    /* Target the entire bottom section */
    .stApp section:last-of-type {
    background: transparent !important;
    }

    /* Override any dark backgrounds in the bottom area */
    .stApp > div > div > div:last-child {
    background: transparent !important;
    }

    /* Target the specific chat input container that appears during chat */
    .stApp > div[data-testid="stAppViewContainer"] > div:last-child,
    .stApp > div[data-testid="stAppViewContainer"] > section:last-child,
    .stApp section[data-testid="stChatInput"],
    .stApp > div > section:last-of-type,
    .stApp > div > div > section:last-child {
        background: transparent !important;
    }

    /* Force any bottom sections to be transparent */
    .stApp section:last-child,
    .stApp section:last-of-type {
        background: transparent !important;
    }

    /* Target any section containing chat elements */
    section:has([data-testid="stChatInput"]) {
        background: transparent !important;
    }

    /* Fallback for any remaining dark backgrounds */
    .stApp [style*="background"] {
        background: transparent !important;
    }

    /* But keep the specific elements we want white */
    .stChatMessage {
    background: white !important;
    }

    .stChatInput > div {
    background: white !important;
    }

    .main .block-container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    }

    /* Keep the main app gradient */
    .stApp {
    background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
    background-attachment: fixed !important;
    }

    /* Fix placeholder text color - must come after the * override */
    .stChatInput input::placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }

    /* Ensure all browser compatibility */
    .stChatInput input::-webkit-input-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    }

    .stChatInput input::-moz-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    }

    .stChatInput input:-ms-input-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    
    }

    /* Target the specific bottom container causing black background */
    .stBottom {
        background: transparent !important;
    }

    .st-emotion-cache-1p2n2i4 {
        background: transparent !important;
    }

    div[data-testid="stBottom"] {
        background: transparent !important;
    }

    /* Target the nested containers */
    .st-emotion-cache-hzygls {
        background: transparent !important;
    }

    div[data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    .st-emotion-cache-6shykm {
        background: transparent !important;
    }

    /* Fix textarea placeholder text color */
    .stChatInput textarea::placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }

    .stChatInput textarea::-webkit-input-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    }

    .stChatInput textarea::-moz-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    }

    .stChatInput textarea:-ms-input-placeholder {
        color: rgba(0, 0, 0, 0.5) !important;
        opacity: 1 !important;
    }

    /* Also target the textarea text color when typing */
    .stChatInput textarea {
        color: #333 !important;
    }

    /* Make chat input scroll with content but keep buttons visible */
    .stChatInput {
        position: relative !important;
        bottom: auto !important;
    }

    /* Target the bottom container */
    .stBottom {
        position: relative !important;
        bottom: auto !important;
    }

    /* Remove fixed positioning from chat input area */
    div[data-testid="stBottom"] {
        position: relative !important;
        bottom: auto !important;
    }

    /* Ensure the input scrolls with the page */
    .st-emotion-cache-1p2n2i4 {
        position: relative !important;
        bottom: auto !important;
    }

    /* Force buttons to be visible */
    .stButton {
        display: block !important;
        visibility: visible !important;
    }

    /* Make sure pills/suggestions are visible */
    .stSelectbox, [data-testid="stSelectbox"] {
        display: block !important;
        visibility: visible !important;
    }
    
</style>
""", unsafe_allow_html=True)


st.html(div(style=styles(font_size=rem(5), line_height=1))["🌱"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Windpark Lindenberg Assistant", anchor=False, width="stretch")

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = "messages" in st.session_state and len(st.session_state.messages) > 0


# Initialize messages with greeting if first time
if not user_first_interaction and not has_message_history:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hallo! 👋 Ich bin ein Informationsassistent für das Windpark Lindenberg Projekt in Beinwil.\n\nSie können mich fragen, was Sie über das Projekt wissen möchten, oder einfach Ihre Gedanken und Sorgen teilen. Ich arbeite nur mit den offiziellen Projektdokumenten und helfe Ihnen dabei, verschiedene Aspekte zu durchdenken.\n\nIch bin neutral - mein Ziel ist es, dass Sie sich besser informiert und vorbereitet fühlen, nicht dass Sie Ihre Meinung ändern. Die Nutzung ist anonym und freiwillig. 🔒\n\nWas geht Ihnen durch den Kopf, wenn Sie an das Windpark-Projekt denken?"}
    ]

# Always display chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        role_name = "assistant"
        avatar = "https://raw.githubusercontent.com/RaffaAmp/ai-supported-reflection-space/refs/heads/main/Avatar.png"
    else:
        role_name = "user"
        avatar = "https://raw.githubusercontent.com/RaffaAmp/ai-supported-reflection-space/refs/heads/main/Avatar_User.png"
    
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(message["content"])
        
# Show input and suggestions only if no interaction yet
if not user_first_interaction and not has_message_history:
    with st.container():
        st.chat_input("Stellen Sie eine Frage...", key="initial_question")
        selected_suggestion = st.pills(
            label="Beispiele",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )
 
    st.button(
    "&nbsp;:small[:gray[:material/info: Über diesen KI-Assistenten]]",
    type="tertiary",
    on_click=show_disclaimer_dialog,
    )
    
    st.stop()
user_message = st.chat_input("Stellen Sie eine Frage...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button("Neustart", icon=":material/refresh:", on_click=clear_conversation)

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user", avatar="https://raw.githubusercontent.com/RaffaAmp/ai-supported-reflection-space/refs/heads/main/Avatar_User.png"):
        st.markdown(user_message)

    with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/RaffaAmp/ai-supported-reflection-space/refs/heads/main/Avatar.png"):
        with st.spinner("Warten..."):
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp

            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

            user_message = user_message.replace("'", "")

        if DEBUG_MODE:
            with st.status("Berechne Prompt...") as status:
                full_prompt = build_question_prompt(user_message)
                st.code(full_prompt)
                status.update(label="Prompt berechnet")
        else:
            with st.spinner("Recherchiere..."):
                full_prompt = build_question_prompt(user_message)

        with st.spinner("Denke nach..."):
            response_gen = get_response(full_prompt)

        with st.container():
            response = st.write_stream(response_gen)
            st.session_state.messages.append({"role": "person", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})
