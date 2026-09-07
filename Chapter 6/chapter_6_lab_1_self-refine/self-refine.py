import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) #keep this line if you don't load_dotenv()
MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 3

def generate_initial_thesis(financials: str) -> str:
    prompt = f"""You are a senior equity analyst at a long-only asset management firm.
Draft a concise investment thesis based on the following financial data:

{financials}

Your thesis should cover key business drivers, valuation, and 2-3 key risks. 
Cite only metrics present in the provided data [Original query]; flag any external claim as [unverified].
Write 1-2 sentences for each section. 
Be concise."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def get_feedback(thesis: str, financials: str, history_feedback: str) -> str:
    prompt = f"""You are a critical investment committee reviewer.
Review the following investment thesis and provide specific,
actionable feedback. Identify missing information, weak arguments, or factual gaps.
Ground your feedback only on the metrics present in the provided data [Original query]; flag any external claim as [unverified].
Your feedback should be based on the main sections present in the thesis: key business drivers, valuation, and 2-3 key risks. 
If no issue would block investment-committee approval, respond with exactly: NO_FURTHER_FEEDBACK
Be concise.

Original query:
{financials}

Previous theses and feedback:
{history_feedback}

Thesis to provide feedback on:
{thesis}


Feedback:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def refine_thesis(thesis: str, feedback: str, financials: str, history_feedback: str) -> str:
    prompt = f"""You are a senior equity analyst.
Revise the following investment thesis based on the reviewer feedback below.
Produce an improved version that directly addresses each point in the feedback. 
Cite only metrics present in the provided data [Original query]; flag any external claim as [unverified].
Your thesis should cover the same sections as the original one: key business drivers, valuation, and 2-3 key risks. 
Write 1-2 sentences for each section.
Be concise.

Original query:
{financials}

Previous theses and feedback:
{history_feedback}

Original Thesis to revise:
{thesis}

Reviewer Feedback:
{feedback}

Revised Thesis:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def self_refine_thesis(financials: str) -> str:
    print("[STEP 1] Generating initial thesis ...")
    thesis = generate_initial_thesis(financials)
    history_feedback = ""
    print(f"Initial Thesis:\n{thesis}\n")
    

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[STEP 2] Iteration {iteration} — Getting feedback...")
        feedback = get_feedback(thesis, financials, history_feedback)
        print(f"Feedback:\n{feedback}\n")    

        if feedback.strip().startswith("NO_FURTHER_FEEDBACK"):
            print("Stopping: model reports no further improvements needed.")
            break
        
        history_feedback += f"Iteration: {iteration}\nThesis: {thesis}\nFeedback: {feedback}\n\n"

        print(f"[STEP 3] Iteration {iteration} — Refining thesis...")
        thesis = refine_thesis(thesis, feedback, financials, history_feedback)
        print(f"Refined Thesis:\n{thesis}\n")

    return thesis, history_feedback

financials = """
----- AAPL -----
Revenue FY2023: $383B (+2% YoY)
Services Revenue: $78B (+16% YoY), Gross Margin ~70%
Net Income: $97B
P/E (forward): 28x | EV/EBITDA: 22x
Cash & Equivalents: $62B | Net Debt: -$52B (net cash)
"""

# Remark: The prompt size grows linearly with each iteration.
final_thesis, history_feedback = self_refine_thesis(financials)
print("FINAL THESIS:\n", final_thesis)
