"""
Job Agent – powered by GPT-4o-mini via OpenRouter (OpenAI-compatible SDK + function calling).

Tools available to the agent:
  1. linkedin_job_search  – searches LinkedIn and returns job listings → auto-saves to CSV
  2. google_search        – web search via Serper API
  3. scrape_website       – scrapes a given URL via FireCrawl

Usage:
  python3 Job_Agent.py
  Results are auto-saved to: scouted_jobs.csv
"""

import os
import csv
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL*", category=Warning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Import the actual tool implementations ──────────────────────────────────
from tools import linkedin_job_search, get_google_search_results, scrape_website

# ── OpenRouter client (OpenAI-compatible) ───────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)
MODEL = "openai/gpt-4o-mini"

# ── CSV output file ──────────────────────────────────────────────────────────
CSV_FILE = "scouted_jobs.csv"
CSV_COLUMNS = [
    "Search Query",
    "Job Title",
    "Company Name",
    "Location",
    "Job Type",
    "Time Posted",
    "Applicants",
    "Job Description",
    "Apply Link",
    "Saved At",
]

# ── Tool schemas (OpenAI function-calling format) ───────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "linkedin_job_search",
            "description": (
                "Search LinkedIn for job postings. Returns listings with title, "
                "company, location, FULL job description, and apply link. "
                "Results are automatically saved to a CSV file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Job title or keywords (e.g. 'Python Developer', 'Data Scientist')."
                    },
                    "location_name": {
                        "type": "string",
                        "description": "City or region to search (e.g. 'India', 'Bangalore', 'Remote')."
                    },
                    "job_type": {
                        "type": "string",
                        "enum": ["onsite", "remote", "hybrid"],
                        "description": "Work arrangement type."
                    },
                    "employment_type": {
                        "type": "string",
                        "enum": ["full-time", "contract", "part-time", "temporary", "internship", "volunteer", "other"],
                        "description": "Employment contract type."
                    },
                    "experience": {
                        "type": "string",
                        "enum": ["internship", "entry-level", "associate", "mid-senior-level", "director", "executive"],
                        "description": "Required experience level."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 5)."
                    },
                    "listed_at": {
                        "type": "integer",
                        "description": "Jobs posted within N seconds (e.g. 86400 = last 24 hours)."
                    },
                    "distance": {
                        "type": "integer",
                        "description": "Distance in miles from the location."
                    }
                },
                "required": ["keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search the web with Google (Serper API). Use for salary research, "
                "company reputation, stipend details, or follow-up questions about "
                "previously listed jobs. Also useful when LinkedIn returns sparse results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Google search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": (
                "Scrape text content of any webpage. Use it to get full job "
                "descriptions, salary info, stipend, or application instructions "
                "from a careers page or job board URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL of the webpage to scrape."
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert AI Job Search Assistant specializing in the Indian job market. "
    "You have PERSISTENT MEMORY of the entire conversation.\n\n"

    "## CRITICAL: NLP Parsing Rules\n\n"
    "### 1. Location (DEFAULT = India)\n"
    "- User mentions a city -> use as location_name.\n"
    "- No location mentioned -> ALWAYS set location_name = 'India'.\n\n"

    "### 2. Company type -> KEYWORDS, NOT location_name\n"
    "  'startup' -> keywords; 'MNC' -> keywords; 'FAANG/MAANG' -> keywords\n\n"

    "### 3. Keyword formula: [job role] + [tech] + [company type]\n"
    "  'internship at startup' -> keywords='internship startup', location_name='India'\n"
    "  '3 Python jobs Bangalore startup' -> keywords='Python developer startup', location_name='Bangalore', limit=3\n\n"

    "### 4. Employment type\n"
    "  'internship'->employment_type='internship' | 'full time'->employment_type='full-time'\n\n"

    "### 5. Limit — 'find me 3' -> limit=3\n\n"

    "## Follow-up questions about listed jobs\n"
    "When user asks about pay/stipend/salary for previously listed jobs:\n"
    "  1. Extract company names from conversation history.\n"
    "  2. Run google_search: '[Company] internship stipend India 2024' or '[Company] salary Glassdoor India'\n"
    "  3. Compare and present findings clearly.\n"
    "  NEVER say you don't have the jobs — you have full conversation history.\n\n"

    "## Search strategy\n"
    "1. ALWAYS call linkedin_job_search first.\n"
    "2. Refine and retry if results look wrong.\n"
    "3. Use google_search only if LinkedIn returns 0 results.\n\n"

    "## Output format (in terminal)\n"
    "**[Job Title]** — [Company] | [Location]\n"
    "Applicants: [count] | Posted: [time] | Type: [employment type]\n"
    "JD Summary: [2-3 sentence summary of the full job description]\n"
    "Apply: [link]\n\n"
    "Always end with: 'Results saved to scouted_jobs.csv'\n\n"
    "Type 'reset' to clear memory."
)


# ── CSV helpers ──────────────────────────────────────────────────────────────
def _init_csv():
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def clean_jd(text: str) -> str:
    """Collapse all newlines/tabs into single spaces so JD fits in one CSV cell."""
    import re
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)   # collapse multiple spaces
    return text.strip()


def save_jobs_to_csv(jobs: list, search_query: str, job_type_filter: str = ""):
    """Append a list of raw job dicts to the CSV file."""
    _init_csv()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows_written = 0

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        for job in jobs:
            writer.writerow({
                "Search Query":    search_query,
                "Job Title":       job.get("job_title", ""),
                "Company Name":    job.get("company_name", ""),
                "Location":        job.get("job_location", ""),
                "Job Type":        job_type_filter or "",
                "Time Posted":     job.get("time_posted", ""),
                "Applicants":      job.get("num_applicants", ""),
                "Job Description": clean_jd(job.get("job_desc_text", "")),
                "Apply Link":      job.get("apply_link", ""),
                "Saved At":        timestamp,
            })
            rows_written += 1

    return rows_written


# ── Tool dispatcher ──────────────────────────────────────────────────────────
# Store the last search query for CSV labelling
_last_search_query = {"value": ""}

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its string output."""
    try:
        if tool_name == "linkedin_job_search":
            raw_jobs = linkedin_job_search(**tool_input)

            if not raw_jobs:
                return "No LinkedIn results found for the given criteria."

            # ── Save FULL JD to CSV ─────────────────────────────────────
            search_q = _last_search_query["value"]
            job_type = tool_input.get("employment_type") or tool_input.get("job_type") or ""
            rows = save_jobs_to_csv(raw_jobs, search_q, job_type)
            print(f"💾 Saved {rows} job(s) to {CSV_FILE}")

            # ── Format for LLM (keep description reasonable length) ─────
            lines = []
            for i, job in enumerate(raw_jobs, 1):
                lines.append(f"--- Job {i} ---")
                lines.append(f"Title:       {job.get('job_title', 'N/A')}")
                lines.append(f"Company:     {job.get('company_name', 'N/A')}")
                lines.append(f"Location:    {job.get('job_location', 'N/A')}")
                lines.append(f"Posted:      {job.get('time_posted', 'N/A')}")
                lines.append(f"Applicants:  {job.get('num_applicants', 'N/A')}")
                lines.append(f"Apply Link:  {job.get('apply_link', 'N/A')}")
                desc = job.get("job_desc_text", "").strip()
                # Give LLM a good chunk to summarize from (800 chars)
                if desc:
                    lines.append(f"Full JD:     {desc[:800]}{'...' if len(desc) > 800 else ''}")
                lines.append("")
            return "\n".join(lines)

        elif tool_name == "google_search":
            result = get_google_search_results.invoke({"query": tool_input["query"]})
            return str(result) if result else "No web results found."

        elif tool_name == "scrape_website":
            result = scrape_website.invoke({"url": tool_input["url"]})
            return str(result) if result else "Could not scrape the website."

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        return f"Tool '{tool_name}' error: {e}"


# ── Persistent conversation history ──────────────────────────────────────────
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


# ── Agentic loop ─────────────────────────────────────────────────────────────
def run_agent(user_query: str):
    """Append user query to persistent history and run the agentic loop."""
    global conversation_history

    # Reset command
    if user_query.lower().strip() in {"new search", "reset", "clear"}:
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("🔄 Conversation reset. Starting fresh!\n")
        return

    # Store the query for CSV labelling
    _last_search_query["value"] = user_query

    # Append user message to running history
    conversation_history.append({"role": "user", "content": user_query})

    print(f"\n🔍 Processing: {user_query}\n")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Add assistant turn to persistent history
        conversation_history.append(msg)

        # ── Final answer ─────────────────────────────────────────────────
        if finish_reason == "stop" or not msg.tool_calls:
            print("🤖", msg.content or "(No response text)")
            break

        # ── Tool calls ───────────────────────────────────────────────────
        if finish_reason == "tool_calls" or msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name  = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                tool_id    = tc.id

                print(f"🔧 Calling: {tool_name}")
                print(f"   Args:   {json.dumps(tool_input, indent=2)}")

                result = dispatch_tool(tool_name, tool_input)
                print(f"📥 Got {len(result)} chars back\n")

                # Append tool result to persistent history
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })


# ── Main REPL ─────────────────────────────────────────────────────────────────
def main():
    _init_csv()  # ensure CSV file exists on startup
    print("╔═══════════════════════════════════════════════════╗")
    print("║         🧑‍💼  AI Job Search Agent  🧑‍💼               ║")
    print("║   GPT-4o-mini · OpenRouter · CSV Export Enabled  ║")
    print("╚═══════════════════════════════════════════════════╝")
    print(f"\n📁 Jobs will be saved to: {os.path.abspath(CSV_FILE)}")
    print("\nExamples:")
    print("  • Find me 3 internships in startups in India")
    print("  • Remote AIML engineer jobs in India")
    print("  • Which of those companies pays the most?")
    print("\nType 'reset' to clear memory | 'exit' to quit.\n")

    while True:
        try:
            user_input = input("👉  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        run_agent(user_input)
        print("\n" + "─" * 55 + "\n")


if __name__ == "__main__":
    main()