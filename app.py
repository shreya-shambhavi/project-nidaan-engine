import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from tools import fetch_pubmed_abstracts
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import gradio as gr

# 1. SETUP & CONFIGURATION

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    temperature = 0
)

# 2. STRUCTURED OUTPUT SCHEMAS (PYDANTIC)

class SynthesisOutput(BaseModel):
    
    key_findings: list[str] = Field(
        description = "A list of 2-3 core medical findings extracted from the abstracts."
    )
    draft_summary: str = Field(
        description = "A single, cohesive paragraph summarizing the overall consensus of the abstracts."
    )

class ReviewerOutput(BaseModel):
    confidence_score: float = Field(
        description = "A score from 0.0 to 1.0 evaluating how accurately the summary reflects the original abstracts."
    )
    feedback: str = Field(
        description = "Specific, actionable feedback on what is missing, inaccurate, or could be improved. If perfect, say 'Looks good.'"
    )

class SearchQueryOutput(BaseModel):
    pubmed_query: str = Field(
        description=(
            "An optimized PubMed search query using MeSH terms and boolean operators. "
            "Example: '(neoplasms[MeSH Terms]) AND (antineoplastic agents[MeSH Terms])'"
        )
    )

class AbstractExtraction(BaseModel):
    pmid: str = Field(description="The PMID from the abstract header.")
    study_type: str = Field(description="RCT, meta-analysis, cohort, case study, or review.")
    sample_size: str = Field(description="Number of participants, or 'Not reported'.")
    population: str = Field(description="The patient group in one sentence.")
    key_finding: str = Field(description="The single most important finding in one sentence.")
    limitation: str = Field(description="Main limitation stated, or 'Not reported'.")

# 3. SHARED GRAPH MEMORY STATE

class AgentState(TypedDict):
    query: str
    pubmed_query: str
    abstracts: List[str]
    extractions: List[dict]
    key_findings: List[str]
    draft_summary: str
    feedback: str
    feedback_history: List[str]
    confidence_score: float
    iterations: int

# 4. MULTI-AGENT GRAPH NODES

def query_planner_node(state: AgentState):
    print("Query Planner Agent: Converting query to PubMed search terms...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a medical librarian expert in PubMed search strategy. "
            "Convert the user's natural language topic into an optimized PubMed query. "
            "Use MeSH terms where appropriate (format: term[MeSH Terms]). "
            "Use AND/OR boolean operators to connect concepts. "
            "Keep the query focused to retrieve high-quality clinical literature. "
            "Example input: 'semaglutide weight loss' "
            "Example output: '(semaglutide[MeSH Terms]) AND "
            "(obesity[MeSH Terms] OR weight reduction[MeSH Terms]) AND (clinical trial[pt])'"
        )),
        ("human", "Convert this topic into a PubMed search query: {query}")
    ])

    structured_llm = llm.with_structured_output(SearchQueryOutput)
    chain = prompt | structured_llm
    result = chain.invoke({"query": state["query"]})

    print(f"Query Planner: Generated → {result.pubmed_query}")
    return {"pubmed_query": result.pubmed_query}

def researcher_node(state: AgentState):
    
    current_iter = state.get("iterations", 0)
    print(f"\n[Iteration {current_iter}] Retriever Agent: Fetching data from PubMed...")
    
    abstracts = fetch_pubmed_abstracts(state["pubmed_query"], max_results = 3)
    return {"abstracts": abstracts, "iterations": current_iter}

def extractor_node(state: AgentState):
    print("Extractor Agent: Pulling structured fields from each abstract...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a medical data extractor. "
            "Extract structured fields from the given PubMed abstract. "
            "Be precise. Extract only what is explicitly stated — do not infer."
        )),
        ("human", "Extract structured fields from this abstract:\n\n{abstract}")
    ])

    structured_llm = llm.with_structured_output(AbstractExtraction)
    chain = prompt | structured_llm

    extractions = []
    for abstract in state["abstracts"]:
        try:
            result = chain.invoke({"abstract": abstract})
            extractions.append(result.model_dump())
        except Exception as e:
            print(f"Extractor Agent: Skipped one abstract — {e}")
            continue

    print(f"Extractor Agent: Extracted fields from {len(extractions)} abstracts.")
    return {"extractions": extractions}

def synthesizer_node(state: AgentState):

    print("Synthesizer Agent: Reviewing data and drafting summary...")
    
    abstracts_text = "\n\n".join(state["abstracts"])

    extractions = state.get("extractions", [])
    extractions_text = "\n\n".join([
        f"Study {i+1} (PMID {ex.get('pmid', 'N/A')}):\n"
        f"  Type: {ex.get('study_type')} | Sample: {ex.get('sample_size')}\n"
        f"  Population: {ex.get('population')}\n"
        f"  Key Finding: {ex.get('key_finding')}\n"
        f"  Limitation: {ex.get('limitation')}"
        for i, ex in enumerate(extractions)
    ])

    feedback = state.get("feedback", "")
    
    system_msg = (
        "You are an expert medical researcher. You are given structured extractions "
        "from each abstract AND the full abstracts for context. "
        "Use the extractions as your primary guide. Write a concise, professional summary."
    )

    feedback_history = state.get("feedback_history", [])

    if feedback_history:
        history_text = "\n".join(feedback_history)
        system_msg += (
                        f"\n\nPREVIOUS DRAFTS WERE REJECTED. "
                        f"YOU MUST ADDRESS ALL OF THE FOLLOWING ISSUES:\n"
                        f"{history_text}\n\n"
                        "Your new draft must resolve every issue listed above, not just the most recent one."
                    )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", (
            "Structured Extractions:\n{extractions}\n\n"
            "Full Abstracts (for context):\n{abstracts}"
        ))
    ])
    
    structured_llm = llm.with_structured_output(SynthesisOutput)
    chain = prompt | structured_llm
    result = chain.invoke({"abstracts": abstracts_text, "extractions": extractions_text})
    
    return {
        "draft_summary": result.draft_summary,
        "key_findings": result.key_findings
    }

def reviewer_node(state: AgentState):
    
    print("Reviewer Agent: Auditing the draft against source materials...")
    
    abstracts_text = "\n\n".join(state["abstracts"])
    draft_summary = state["draft_summary"]

    key_findings = state.get("key_findings", [])
    findings_text = "\n".join(
        [f"{i+1}. {f}" for i, f in enumerate(key_findings)]
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a medical reviewer with three tasks:\n"
            "1. Check that every Extracted Key Finding is accurately reflected in the Draft Summary.\n"
            "2. Check that the Draft Summary contains no claims absent from the Original Abstracts.\n"
            "3. Assign a confidence score (0.0–1.0). 0.9+ = accurate and complete. "
            "0.8 = minor gaps. Below 0.85 = requires revision.\n"
            "Provide specific, actionable feedback only if score is below 0.85."

            # "You are a medical reviewer evaluating a summary. "
            # "Compare the Draft against the Original Abstracts. "
            # "Give a score (0.0 to 1.0). "
            # "Scoring guide: 0.9+ means accurate and concise. 0.8 means good but minor flaws. "
            # "Provide specific feedback only if the score is below 0.85."
            # "You are a strict, highly critical medical peer reviewer. "
            # "Compare the Draft Summary against the Original Abstracts. "
            # "Does the summary capture the nuance? Are there any false claims? "
            # "Give a confidence score (0.0 to 1.0) and brutal, actionable feedback."
        )),
        ("human", "Original Abstracts:\n{abstracts}\n\nKey Findings:\n{findings}\n\nDraft Summary:\n{summary}")
    ])
    
    structured_llm = llm.with_structured_output(ReviewerOutput)
    chain = prompt | structured_llm
    result = chain.invoke({"abstracts": abstracts_text, "findings": findings_text, "summary": draft_summary})
    
    current_iter = state["iterations"] + 1
    history = state.get("feedback_history", [])

    if result.feedback and result.feedback.strip().lower() != "looks good.":
        entry = f"Iteration {current_iter}: {result.feedback}"
        history = history + [entry]

    return {
        "confidence_score": result.confidence_score,
        "feedback": result.feedback,
        "feedback_history": history,
        "iterations": current_iter
    }

# 5. CONDITIONAL EDGE ROUTING

def route_after_review(state: AgentState):
    
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iterations", 0)
    
    if score >= 0.85:
        print(f"Router: Quality score ({score}) meets requirements. Exiting graph.")
        return "end"
    elif iterations >= 3:
        print(f"Router: Reached maximum limits ({iterations}/3 loops). Forcing safety exit.")
        return "end"
    else:
        print(f"Router: Score ({score}) is unsatisfactory. Re-routing back to Synthesizer.")
        return "continue"

# 6. GRAPH ORCHESTRATION BUILDER

builder = StateGraph(AgentState)

builder.add_node("query_planner", query_planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("extractor", extractor_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "query_planner")
builder.add_edge("query_planner", "researcher")
builder.add_edge("researcher", "extractor")
builder.add_edge("extractor", "synthesizer")
builder.add_edge("synthesizer", "reviewer")

builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "continue": "synthesizer",  
        "end": END                  
    }
)

graph = builder.compile()

# 7. GRADIO WEB INTERFACE

def run_agent_pipeline(user_query: str):

    if not user_query.strip():
        return "Please enter a valid medical topic.", "", "", [], "", 0.0, 0

    try:
        final_output = graph.invoke({"query": user_query, "iterations": 0})

        summary      = final_output.get("draft_summary", "No summary generated.")
        score        = round(final_output.get("confidence_score", 0.0), 2)
        iterations   = final_output.get("iterations", 0)
        pubmed_query = final_output.get("pubmed_query", "N/A")
        key_findings = final_output.get("key_findings", [])
        extractions  = final_output.get("extractions", [])
        history      = final_output.get("feedback_history", [])

        findings_text = "\n".join(
            f"{i + 1}. {f}" for i, f in enumerate(key_findings)
        ) or "No key findings extracted."

        extraction_rows = [
            [
                ex.get("pmid",        "N/A"),
                ex.get("study_type",  "N/A"),
                ex.get("sample_size", "N/A"),
                ex.get("population",  "N/A"),
                ex.get("key_finding", "N/A"),
                ex.get("limitation",  "N/A"),
            ]
            for ex in extractions
        ]

        history_text = (
            "\n\n".join(history) if history
            else "Draft passed reviewer on first attempt."
        )

        return summary, pubmed_query, findings_text, extraction_rows, history_text, score, iterations

    except Exception as e:
        return f"Pipeline Error: {str(e)}", "", "", [], "", 0.0, 0


with gr.Blocks(title="Project Nidaan Engine") as demo:

    gr.Markdown("""
    # 🧬 Project Nidaan Engine
    **Multi-agent biomedical literature synthesizer** &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp;
    Llama 3.3 70B via Groq &nbsp;·&nbsp; PubMed NCBI

    Five agents collaborate in sequence: a **Query Planner** optimizes your search,
    a **Retriever** fetches PubMed abstracts, an **Extractor** pulls structured fields,
    a **Synthesizer** drafts a consensus summary, and a **Critic** scores and re-routes
    until quality passes threshold.
    """)

    # ── Input row ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Medical Research Topic",
                placeholder="e.g., efficacy of semaglutide in type 2 diabetes management",
                lines=2,
            )
            submit_btn = gr.Button("🚀 Deploy Agents", variant="primary", size="lg")

        with gr.Column(scale=1):
            score_output = gr.Number(label="Reviewer Confidence (0–1)", precision=2)
            iter_output  = gr.Number(label="Revision Loops Taken",      precision=0)

    # ── Primary output ─────────────────────────────────────────────────────────
    summary_output = gr.Textbox(
        label="Final Peer-Reviewed Summary",
        lines=10,
        interactive=False,
    )

    # ── Pipeline trace (collapsed by default) ─────────────────────────────────
    with gr.Accordion("🔬 Pipeline Trace — expand to inspect agent reasoning", open=False):

        pubmed_query_output = gr.Textbox(
            label="Optimized PubMed Query  [Query Planner Agent]",
            interactive=False,
        )

        findings_output = gr.Textbox(
            label="Key Findings  [Synthesizer Agent]",
            lines=4,
            interactive=False,
        )

        extractions_output = gr.DataFrame(
            label="Structured Extractions per Abstract  [Extractor Agent]",
            headers=[
                "PMID", "Study Type", "Sample Size",
                "Population", "Key Finding", "Limitation",
            ],
            interactive=False,
            wrap=True,
        )

        history_output = gr.Textbox(
            label="Reviewer Feedback History  [Critic Agent]",
            lines=4,
            interactive=False,
        )

    # ── Wire outputs ───────────────────────────────────────────────────────────
    outputs = [
        summary_output,
        pubmed_query_output,
        findings_output,
        extractions_output,
        history_output,
        score_output,
        iter_output,
    ]

    submit_btn.click(fn=run_agent_pipeline, inputs=query_input, outputs=outputs)
    query_input.submit(fn=run_agent_pipeline, inputs=query_input, outputs=outputs)

if __name__ == "__main__":
    print("Launching Nidaan Engine...")
    demo.launch(share=True)
