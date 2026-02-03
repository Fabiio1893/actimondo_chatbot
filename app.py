# app.py
# ---------------------------------------------------------
# Streamlit GraphRAG Chatbot für einen Neo4j Aura Wissensgraphen
# (dynamische Labels/Prädikate; CSV-Header bleiben stabil)
#
# Erwartetes Graph-Schema (aus deinem Cypher-Import abgeleitet):
# - Nodes: dynamische Labels (row.subjectType / row.objectType)
#          Properties: key (unique), name, normalizedKey (optional), qualifiers (optional)
# - Relationships: type = row.predicate
#          Properties: sourceDoc, sourceSection, sourcePage (int/nullable),
#                      sourceQuote (optional), confidence (optional)
# ---------------------------------------------------------

import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

# Optional: nur für lokale Entwicklung
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ----------------------------
# Config Loader (Secrets -> ENV)
# ----------------------------
def get_cfg(key: str, default=None):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


NEO4J_URI      = get_cfg("NEO4J_URI")
NEO4J_USER     = get_cfg("NEO4J_USER")
NEO4J_PASSWORD = get_cfg("NEO4J_PASSWORD")
NEO4J_DATABASE = get_cfg("NEO4J_DATABASE", "neo4j")

OPENAI_MODEL   = get_cfg("OPENAI_MODEL", "gpt-4.1-mini")


# ----------------------------
# Cached Clients
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


# ----------------------------
# Neo4j Helper
# ----------------------------
def cypher_query(driver, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    params = params or {}
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, params)
        return [r.data() for r in res]


# ----------------------------
# LLM Helper
# ----------------------------
def llm_json(prompt: str) -> Any:
    """Call LLM and parse JSON output safely."""
    client = get_openai_client(st.session_state["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
        if not m:
            return None
        return json.loads(m.group(0))


def extract_search_terms(question: str) -> List[str]:
    """
    Extrahiert Suchbegriffe (nicht nur 'Concept'), damit der Bot mit beliebigen Ontologien skaliert.
    """
    prompt = f"""
Extrahiere aus der Frage kurze Suchbegriffe, die wahrscheinlich als Node-Namen im Wissensgraphen vorkommen.
- Gib eine JSON-Liste von Strings aus
- max 10 Einträge
- keine Erklärungen
- eher Nomen/Eigennamen/Begriffe/Abschnitte

Frage: {question}
"""
    data = llm_json(prompt)
    if isinstance(data, list):
        out = []
        for x in data:
            s = str(x).strip()
            if s and s.lower() not in {"null", "none"}:
                out.append(s)
        return out[:10]
    return []


# ----------------------------
# GraphRAG Retrieval
# ----------------------------
def seed_nodes(driver, terms: List[str], limit_per_term: int = 6) -> List[Dict[str, Any]]:
    """
    Findet Startknoten flexibel über name/key/normalizedKey.
    Returns: List[{key,name,labels}]
    """
    seeds: Dict[str, Dict[str, Any]] = {}

    q = """
    MATCH (n)
    WITH n,
        toLower(n.name) AS nameL,
        toLower(n.key) AS keyL,
        toLower(n.normalizedKey) AS normL
    WHERE (n.name IS NOT NULL AND (nameL = toLower($t) OR nameL CONTAINS toLower($t)))
    OR (n.key IS NOT NULL AND keyL CONTAINS toLower($t))
    OR (n.normalizedKey IS NOT NULL AND (normL = toLower($t) OR normL CONTAINS toLower($t)))
    RETURN n.key AS key, n.name AS name, labels(n) AS labels
    LIMIT $lim
    """


    for t in terms:
        rows = cypher_query(driver, q, {"t": t, "lim": limit_per_term})
        for r in rows:
            k = r.get("key")
            if k:
                seeds[k] = r

    return list(seeds.values())


def expand_subgraph(driver, seed_keys: List[str], hops: int = 2, max_triples: int = 250) -> List[Dict[str, Any]]:
    """
    Expandiere um Seed-Nodes (undirected in der Expansion), liefere danach gerichtete Tripel zurück.
    """
    if not seed_keys:
        return []

    q = f"""
    MATCH (s)
    WHERE s.key IN $seed_keys

    MATCH (s)-[rs*1..{hops}]-(t)
    WITH rs
    UNWIND rs AS rel
    WITH DISTINCT rel

    MATCH (a)-[rel]->(b)
    RETURN
      a.key   AS sKey,
      a.name  AS sName,
      labels(a) AS sLabels,
      type(rel) AS p,
      b.key   AS oKey,
      b.name  AS oName,
      labels(b) AS oLabels,
      rel.sourceDoc     AS sourceDoc,
      rel.sourceSection AS sourceSection,
      rel.sourcePage    AS sourcePage,
      rel.sourceQuote   AS sourceQuote,
      rel.confidence    AS confidence
    LIMIT $max_triples
    """
    return cypher_query(driver, q, {"seed_keys": seed_keys, "max_triples": max_triples})


def _tokenize(text: str) -> set:
    return set(re.findall(r"[A-Za-zÄÖÜäöüß0-9_]+", (text or "").lower()))


def score_triple(triple: Dict[str, Any], question: str) -> float:
    q_tokens = _tokenize(question)

    s = triple.get("sName") or ""
    o = triple.get("oName") or ""
    p = triple.get("p") or ""
    sl = " ".join(triple.get("sLabels") or [])
    ol = " ".join(triple.get("oLabels") or [])
    quote = triple.get("sourceQuote") or ""

    text = f"{s} {p} {o} {sl} {ol} {quote}".lower()
    t_tokens = _tokenize(text)
    overlap = len(q_tokens & t_tokens)

    return overlap / (len(q_tokens) + 1e-9)


def rank_triples(triples: List[Dict[str, Any]], question: str, k: int = 30) -> List[Dict[str, Any]]:
    scored = [(score_triple(t, question), t) for t in triples]
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [t for s, t in scored[:k] if s > 0]
    if top:
        return top

    return [t for _, t in scored[:k]]


def build_context(triples: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Erzeuge kompakten Kontext mit Quellen und optionalem Quote.
    """
    lines = []
    sources = []

    for t in triples:
        s = t.get("sName") or t.get("sKey") or "?"
        o = t.get("oName") or t.get("oKey") or "?"
        p = t.get("p") or "?"

        s_labels = t.get("sLabels") or []
        o_labels = t.get("oLabels") or []

        doc = t.get("sourceDoc")
        sec = t.get("sourceSection")
        page = t.get("sourcePage")
        quote = t.get("sourceQuote")
        conf = t.get("confidence")

        cite_parts = []
        if doc:
            cite_parts.append(str(doc))
        if sec:
            cite_parts.append(str(sec))
        if page is not None and str(page) != "":
            cite_parts.append(f"S.{page}")
        cite = " | ".join(cite_parts) if cite_parts else "Quelle: unbekannt"

        if cite not in sources:
            sources.append(cite)

        label_str = ""
        if s_labels or o_labels:
            label_str = f" ({'/'.join(s_labels)} → {'/'.join(o_labels)})"

        conf_str = f" [conf: {conf}]" if conf else ""
        quote_str = f' — "{quote}"' if quote else ""

        lines.append(f"- {s} —[{p}]→ {o}{label_str}{conf_str}  ({cite}){quote_str}")

    return "\n".join(lines), sources


# ----------------------------
# Answering
# ----------------------------
def answer_with_rag(question: str, context: str) -> str:
    client = get_openai_client(st.session_state["OPENAI_API_KEY"])

    system = (
        "Du bist ein wissensbasierter Assistent.\n"
        "Antworte ausschließlich basierend auf dem gegebenen Kontext aus einem Neo4j-Wissensgraphen.\n"
        "Wenn die Information im Kontext nicht enthalten ist, sage klar: 'Nicht im Wissensgraph abgedeckt'.\n"
        "Zitiere die Quelle(n) aus dem Kontext (Dokument/Abschnitt/Seite).\n"
    )

    user = f"""
KONTEXT (Tripel mit Quellen):
{context}

FRAGE:
{question}

AUFGABE:
- Gib eine präzise, strukturierte Antwort.
- Beziehe dich nur auf den Kontext.
- Füge am Ende 'Quellen:' mit den verwendeten Quellen (Dokument | Abschnitt | Seite) hinzu.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="GraphRAG Chatbot (Neo4j)", page_icon="🧠", layout="wide")
st.title("🧠 GraphRAG Chatbot (Neo4j Aura)")

with st.sidebar:
    st.markdown("### OpenAI")
    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("OPENAI_API_KEY", default_key),
        help="Wird nur in dieser Session gespeichert.",
    )
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
    else:
        st.warning("Bitte OpenAI API Key eingeben, sonst kann der Chatbot nicht antworten.")
        st.stop()

    st.markdown("---")
    st.markdown("### Neo4j Verbindung")
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        st.error("Neo4j Zugangsdaten fehlen. Bitte in Streamlit Secrets/ENV setzen.")
        st.stop()

    st.write("Neo4j URI:", NEO4J_URI)
    st.write("Neo4j User:", NEO4J_USER)
    st.write("Neo4j DB:", NEO4J_DATABASE)
    st.write("LLM Model:", OPENAI_MODEL)

    st.markdown("---")
    hops = st.slider("Graph-Expansion (Hops)", 1, 4, 2)
    topk = st.slider("Top-K Tripel für Kontext", 5, 60, 30)

    st.markdown("---")
    debug = st.checkbox("🪲 Debug-Ausgaben anzeigen", value=False)



# Driver + Connection-Test
driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
try:
    cypher_query(driver, "RETURN 1 AS ok")
except Exception as e:
    st.error(f"Neo4j Verbindung fehlgeschlagen: {e}")
    st.stop()

if "debug" not in st.session_state:
    st.session_state["debug"] = False
st.session_state["debug"] = debug

if debug:
    stats = cypher_query(driver, "MATCH (n) RETURN count(n) AS nodeCount")
    stats2 = cypher_query(driver, "MATCH ()-[r]->() RETURN count(r) AS relCount")
    st.sidebar.write("Nodes:", stats[0]["nodeCount"] if stats else "n/a")
    st.sidebar.write("Rels:", stats2[0]["relCount"] if stats2 else "n/a")


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Stell eine Frage zum Wissensgraphen…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("GraphRAG sucht im Neo4j-Graph…"):
            terms = extract_search_terms(question)

            # Fallback: einfache Keywords, falls LLM nichts liefert
            if not terms:
                terms = list({w for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]+", question) if len(w) > 4})[:8]

            seeds = seed_nodes(driver, terms)
            seed_keys = [s["key"] for s in seeds if s.get("key")]

            # Noch ein robuster Fallback: wenn Seeds leer, probiere die längsten Wörter
            if not seed_keys:
                fallback_terms = sorted(
                    {w for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]+", question) if len(w) > 5},
                    key=len,
                    reverse=True,
                )[:8]
                seeds = seed_nodes(driver, fallback_terms)
                seed_keys = [s["key"] for s in seeds if s.get("key")]

            triples = expand_subgraph(driver, seed_keys, hops=hops) if seed_keys else []
            ranked = rank_triples(triples, question, k=topk)
            context, _sources = build_context(ranked)

            if not context.strip():
                answer = "Nicht im Wissensgraph abgedeckt."
            else:
                answer = answer_with_rag(question, context)

        st.markdown(answer)

        with st.expander("🔎 Verwendeter Graph-Kontext"):
            st.markdown(context if context else "_Kein Kontext gefunden._")

    st.session_state.messages.append({"role": "assistant", "content": answer})
