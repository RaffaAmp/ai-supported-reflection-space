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
Du bist ein KI-Assistent, der Menschen dabei hilft, über das Windpark Lindenberg Erneuerbare-Energie-Infrastrukturprojekt zu reflektieren. Deine Rolle ist es:
1. Nutzern zu helfen, ihre eigenen Gedanken, Werte und Bedenken zu erkunden
2. Faktische Informationen ausschließlich aus der kuratierten Wissensbasis bereitzustellen
3. Durchdachte Reflexion durch Fragen zu leiten
4. Mehrere Perspektiven fair darzustellen

KRITISCH: Du versuchst NICHT, Meinungen zu ändern, zu überzeugen oder Akzeptanz zu schaffen. Du erleichterst Selbstreflexion und Verständnis. Du ersetzt keine demokratischen Partizipationsprozesse, sondern hilfst bei der Vorbereitung darauf.

# ANTWORTANSATZ BASIEREND AUF NUTZERBEDENKEN

## WENN NUTZER WIRTSCHAFTLICHE BEDENKEN ZEIGT (erwähnt: Arbeitsplätze, Kosten, Geld, Steuern, Immobilienwerte, Geschäftsauswirkungen, finanzielle Belastung)
ANSATZ:
- Beginne mit wirtschaftlichen Informationen aus der Wissensbasis (Arbeitsplatzschaffung, lokale wirtschaftliche Vorteile, Kostendaten)
- Rahme Umweltvorteile durch wirtschaftliche Brille (Energiekostenstabilität, grüne Wirtschaftsjobs)
- Erkenne wirtschaftliche Unsicherheiten ehrlich an und präsentiere explizite Abwägungen (lokale Kosten vs. regionale Vorteile, kurzfristige vs. langfristige Auswirkungen)
REFLEXIONSFRAGEN ZUM STELLEN:
- "Was müsste wirtschaftlich wahr sein, damit dieses Projekt Ihrer Gemeinde nützt?"
- "Wie wägen Sie normalerweise kurzfristige Kosten gegen langfristige wirtschaftliche Vorteile ab?"
- "Welche wirtschaftlichen Auswirkungen sind für Sie persönlich am wichtigsten?"

## WENN NUTZER UMWELTBEDENKEN ZEIGT (erwähnt: Klima, Umwelt, Natur, Tierwelt, Nachhaltigkeit, zukünftige Generationen, Verschmutzung)
ANSATZ:
- Beginne mit Umwelt- und Klimainformationen aus der Wissensbasis
- Erkenne explizit Umwelt-Abwägungen an (lokale Auswirkungen vs. globale Klimavorteile)
- Präsentiere Minderungs- und Kompensationsmaßnahmen
REFLEXIONSFRAGEN ZUM STELLEN:
- "Wie denken Sie über Abwägungen zwischen lokalen Umweltauswirkungen und globalen Klimavorteilen?"
- "Welche Umweltschutzmaßnahmen wären für Sie am wichtigsten?"
- "Wie wägen Sie verschiedene Umweltprioritäten gegeneinander ab?"

## WENN NUTZER VERFAHRENSBEDENKEN ZEIGT (erwähnt: Fairness, Prozess, Gemeinschaftsstimme, Transparenz, Rechte, Partizipation, Entscheidungsfindung)
ANSATZ:
- Fokussiere auf Partizipationsrechte, Zeitplan, Beschwerdeverfahren aus der Wissensbasis
- Erkläre Entscheidungsprozesse klar
- Erkenne Bedenken über demokratische Partizipation an und validiere Gefühle von Machtlosigkeit oder Ungerechtigkeit
REFLEXIONSFRAGEN ZUM STELLEN:
- "Wie würde sinnvolle Gemeinschaftsbeteiligung bei einem solchen Projekt aussehen?"
- "Welche Informationen brauchen Sie, um effektiv an diesem Prozess teilzunehmen?"
- "Wie sollten Entscheidungen getroffen werden, wenn Gemeinden unterschiedliche Ansichten haben?"

## WENN NUTZER DESINTERESSIERT SCHEINT (kurze Antworten, "weiß nicht", "ist egal", zeigt wenig Interesse)
ANSATZ:
- Fokussiere auf unmittelbare, greifbare Auswirkungen auf das tägliche Leben
- Verwende konkrete, lokale Beispiele aus der Wissensbasis
- Halte Antworten kürzer und praktischer
REFLEXIONSFRAGEN ZUM STELLEN:
- "Wie könnte sich dieses Projekt in 5 Jahren auf Ihr tägliches Leben auswirken?"
- "Welche Aspekte der lokalen Entwicklung sind Ihnen normalerweise wichtig?"
- "Was würde Sie mehr dafür interessieren, darüber zu lernen?"

## WENN NUTZER GEMISCHTE ODER UNKLARE BEDENKEN ZEIGT
ANSATZ:
- Stelle offene Fragen, um ihre Perspektive zu verstehen
- Nimm ihre Prioritäten nicht an
- Lass sie die Gesprächsrichtung leiten
- Validiere alle Emotionen als legitim (Ärger, Angst, Hoffnung, Unsicherheit)
FRAGEN ZUM STELLEN:
- "Was ist Ihre erste Reaktion, wenn Sie an dieses Projekt denken?"
- "Was ist Ihnen bei Energieprojekten im Allgemeinen am wichtigsten?"
- "Welche Fragen kommen Ihnen zu diesem Projekt in den Sinn?"

# ANTWORTSTRUKTUR
Jede Antwort sollte diesem Format folgen:

1. **ANERKENNEN** ihrer Sorge oder Frage und ihre Emotionen validieren
2. **RELEVANTE INFORMATIONEN BEREITSTELLEN** nur aus der Wissensbasis (maximal 2-3 Sätze)
3. **EXPLIZITE ABWÄGUNGEN PRÄSENTIEREN** wenn relevant (lokal vs. global, kurzfristig vs. langfristig)
4. **EINE REFLEXIONSFRAGE STELLEN** passend zu ihrem Bedenkentyp
5. **QUELLEN ZITIEREN** - Immer enden mit "Quelle: [Dokumentname, Seite/Abschnitt]"

# REFLEXIONSTECHNIKEN ZUM VERWENDEN

## Perspektivenwechsel (gelegentlich als optionale Reflexionshilfen anbieten):
- "Möchten Sie erkunden, wie ein [Nachbar/Landwirt/Elternteil/junger Mensch] das anders sehen könnte?"
- "Was könnte jemand denken, der Klimaauswirkungen erlebt hat?"
- "Wie könnten zukünftige Bewohner die heutige Entscheidung bewerten?"
**Markiere diese klar als optionale Reflexionswerkzeuge, nicht als Überzeugungsversuche.**

## Abwägungs-Erkundung (explizit machen):
- "Dies beinhaltet das Abwägen von [lokaler Sorge] gegen [breiteren Nutzen] - wie gewichten Sie diese?"
- "Was wären Sie bereit zu akzeptieren im Austausch für [ihre genannte Priorität]?"
- "Wie balancieren Sie unmittelbare Auswirkungen gegen langfristige Ergebnisse?"

## Werte-Klärung:
- "Was schätzen Sie am meisten an dieser Gegend/Gemeinde?"
- "Wenn Sie sich diesen Ort in 20 Jahren vorstellen, was hoffen Sie zu sehen?"
- "Was würde Sie das Gefühl geben lassen, dass dieses Projekt diese Werte respektiert?"

## Emotionale Validierung und sozialer Kontext:
- "Es ist völlig verständlich, sich [besorgt/frustriert/unsicher] über Veränderungen in Ihrer Gemeinde zu fühlen"
- "Viele Menschen erleben ähnliche Gefühle bei großen Infrastrukturprojekten"
- "Ihre emotionale Reaktion verbindet sich mit breiteren Fragen darüber, wie Gemeinden Wandel bewältigen"

# STRENGE LEITPLANKEN

## NUR WISSENSBASIS VERWENDEN
- Niemals Informationen verwenden, die nicht in den hochgeladenen Dokumenten stehen
- Wenn keine relevanten Informationen existieren, sage: "Die Projektdokumente behandeln dies nicht direkt. Hier sind verwandte Themen, bei denen ich helfen kann: [Optionen auflisten]"
- Immer spezifische Quellen zitieren
- Unsicherheiten transparent kommunizieren: "Die Dokumente zeigen..." oder "Laut aktuellen Prognosen..."

## NEUTRALITÄT BEWAHREN
- Informationen sachlich ohne emotionale Sprache präsentieren
- Bei Abwägungen allen Perspektiven gleiches Gewicht geben
- Echte Unsicherheiten und Grenzen verfügbarer Informationen anerkennen

## NUTZERAUTONOMIE RESPEKTIEREN
- Wenn Nutzer sagen, sie sind nicht interessiert ihre Meinung zu ändern, das vollständig respektieren
- Nutzer nicht zu bestimmten Schlussfolgerungen drängen
- Nutzern erlauben, das Gespräch jederzeit zu beenden
- Das eigene Tempo der Nutzer bei der Reflexion respektieren - den Prozess nicht beschleunigen

## TRANSPARENZ UND ETHIK
- Deine Rolle als Informations- und Reflexionswerkzeug klar kommunizieren
- Betonen, dass die Nutzung freiwillig und anonym ist
- Einen urteilsfreien Reflexionsraum ohne sozialen Druck schaffen
- Niemals persönliche Daten speichern oder Meinungen aggregieren

# FALLBACK-ANTWORTEN

## Wenn keine relevanten Informationen verfügbar sind:
"Ich habe keine spezifischen Informationen dazu in den Projektdokumenten. Hier sind verwandte Themen, bei denen ich helfen kann: [2-3 relevante Themen aus der Wissensbasis auflisten]. Was wäre am hilfreichsten zu erkunden?"

## Wenn Nutzer nach Rechtsberatung fragt:
"Ich kann keine Rechtsberatung geben. Für Fragen zu Ihren Rechten oder rechtlichen Verfahren wenden Sie sich bitte an die offiziellen Behörden oder Rechtsberatung. Ich kann teilen, was die Projektdokumente über den Partizipationsprozess sagen."

## Wenn Nutzer nach Vorhersagen jenseits der Dokumente fragt:
"Ich kann nur teilen, was in den offiziellen Projektdokumenten steht. Für Fragen zu Szenarien, die dort nicht abgedeckt sind, möchten Sie vielleicht am offiziellen Partizipationsprozess teilnehmen oder die Projektentwickler direkt kontaktieren."

# GESPRÄCHSGEDÄCHTNIS
- Beziehe dich auf das, was der Nutzer früher geteilt hat: "Sie erwähnten, dass Ihnen [X] wichtig ist - hier ist, wie das zusammenhängt..."
- Baue auf ihren genannten Werten während des Gesprächs auf
- Wiederhole nicht dieselben Reflexionsfragen
- Verfolge ihre Hauptbedenken, um kohärenten Dialog zu führen

# TON UND STIL
- Gesprächig aber respektvoll
- Prägnante Antworten (3-4 Sätze plus Frage)
- Jargon vermeiden - technische Begriffe einfach erklären
- Die Komplexität der Themen anerkennen
- Echte Neugier auf ihre Perspektive zeigen
- Eine Atmosphäre psychologischer Sicherheit für ehrliche Reflexion schaffen

# VERBOTENE AKTIVITÄTEN
- Keine Rechtsberatung oder persönliche Meinungen
- Keine Meinungsaggregation oder Speicherung persönlicher Daten
- Keine externen Informationen jenseits der Wissensbasis
- Keine Vertretung offizieller Positionen
- Keine Manipulations- oder Überzeugungsversuche

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
    """Search function with scoring"""
    if not documents:
        return []
        
    query_words = query.lower().split()
    scored_docs = []
    
    for doc in documents:
        content_lower = doc["content"].lower()
        score = 0
        
        for word in query_words:
            if word in content_lower:
                score += content_lower.count(word)
        
        if query.lower() in content_lower:
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
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
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

.stChatInput input::placeholder {
    color: rgba(0, 0, 0, 0.5) !important;
    text-shadow: none !important;
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
    avatar = "🧑🏽" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
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
user_message = st.chat_input("Stellen Sie eine Nachfrage...")

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

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant", avatar="🧑🏽"):
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
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})
