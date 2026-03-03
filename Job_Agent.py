"""
Job Agent – powered by GPT-4o-mini via OpenRouter (OpenAI-compatible SDK + function calling).

Tools available to the agent:
  1. linkedin_job_search  – searches LinkedIn and returns job listings
  2. google_search        – web search via Serper API
  3. scrape_website       – scrapes a given URL via FireCrawl

Usage:
  python3 Job_Agent.py
"""

import os
import json
import warnings

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

# ── Tool schemas (OpenAI function-calling format) ───────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "linkedin_job_search",
            "description": (
                "Search LinkedIn for job postings. Returns listings with title, "
                "company, location, description, and apply link."
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
                "Search the web with Google (Serper API). Useful for finding extra "
                "job posts on company career pages or getting company/role context."
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
                "descriptions, salary info, or application instructions from "
                "a careers page or job board URL."
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
    "You are an expert AI Job Search Assistant specializing in the Indian job market.\n\n"

    "## CRITICAL: NLP Parsing Rules — read BEFORE calling any tool\n\n"

    "### 1. Location (DEFAULT = India)\n"
    "- User mentions a city (Bangalore, Mumbai, Delhi, Hyderabad, Pune) -> use it as location_name.\n"
    "- User says 'India' -> location_name = 'India'.\n"
    "- User mentions NO location at all -> ALWAYS set location_name = 'India'. Never leave it empty.\n\n"

    "### 2. Company type words -> go into KEYWORDS, NOT location_name\n"
    "These describe company type. Add them to the keywords field, never to location_name:\n"
    "  'startup' -> add 'startup' to keywords\n"
    "  'MNC' or 'multinational' -> add 'MNC' to keywords\n"
    "  'product company' -> add 'product company' to keywords\n"
    "  'FAANG' or 'MAANG' -> add 'FAANG' to keywords\n"
    "  'agency' or 'consultancy' -> add to keywords\n\n"

    "### 3. Keyword construction formula\n"
    "[job role] + [tech/domain] + [company type if mentioned]\n"
    "Examples:\n"
    "  'internship at startup in india' -> keywords='internship startup', location_name='India', employment_type='internship'\n"
    "  '3 Python jobs in Bangalore startup' -> keywords='Python developer startup', location_name='Bangalore', limit=3\n"
    "  'remote ML engineer' -> keywords='machine learning engineer', job_type='remote', location_name='India'\n"
    "  'data science internship today' -> keywords='data scientist intern', employment_type='internship', listed_at=86400, location_name='India'\n\n"

    "### 4. Employment type\n"
    "  'internship' -> employment_type='internship'\n"
    "  'full time' -> employment_type='full-time'\n"
    "  'contract' -> employment_type='contract'\n"
    "  'part time' -> employment_type='part-time'\n\n"

    "### 5. Limit\n"
    "If user says 'find me 3' or 'show 5' -> set limit to that number.\n\n"

    "## Search Strategy\n"
    "1. ALWAYS call linkedin_job_search first with correctly parsed params.\n"
    "2. If results look wrong (wrong country, irrelevant roles), call it again with refined keywords.\n"
    "3. Use google_search only if LinkedIn returns 0 results.\n\n"

    "## Output Format\n"
    "**[Job Title]** — [Company] | [Location]\n"
    "Applicants: [count if available] | Posted: [time]\n"
    "Description: [1-2 sentences]\n"
    "Apply: [link]\n\n"
    "If results seem off-target, say so clearly and offer to refine."
)

# ── Tool dispatcher ──────────────────────────────────────────────────────────
def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its string output."""
    try:
        if tool_name == "linkedin_job_search":
            result = linkedin_job_search(**tool_input)
            if not result:
                return "No LinkedIn results found for the given criteria."
            lines = []
            for i, job in enumerate(result, 1):
                lines.append(f"--- Job {i} ---")
                lines.append(f"Title:      {job.get('job_title', 'N/A')}")
                lines.append(f"Company:    {job.get('company_name', 'N/A')}")
                lines.append(f"Location:   {job.get('job_location', 'N/A')}")
                lines.append(f"Posted:     {job.get('time_posted', 'N/A')}")
                lines.append(f"Applicants: {job.get('num_applicants', 'N/A')}")
                lines.append(f"Apply Link: {job.get('apply_link', 'N/A')}")
                desc = job.get("job_desc_text", "")
                if desc:
                    lines.append(f"Desc:       {desc[:400]}{'...' if len(desc) > 400 else ''}")
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


# ── Agentic loop ─────────────────────────────────────────────────────────────
def run_agent(user_query: str):
    """Run the agentic tool-use loop for one user query."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    print(f"\n🔍 Searching: {user_query}\n")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Add assistant message to history
        messages.append(msg)

        # ── Final answer ─────────────────────────────────────────────────────
        if finish_reason == "stop" or not msg.tool_calls:
            print("🤖", msg.content or "(No response text)")
            break

        # ── Tool calls ───────────────────────────────────────────────────────
        if finish_reason == "tool_calls" or msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name  = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                tool_id    = tc.id

                print(f"🔧 Calling: {tool_name}")
                print(f"   Args:   {json.dumps(tool_input, indent=2)}")

                result = dispatch_tool(tool_name, tool_input)

                print(f"📥 Got {len(result)} chars back\n")

                # Feed the tool result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })


# ── Main REPL ─────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════╗")
    print("║       🧑‍💼  AI Job Search Agent  🧑‍💼        ║")
    print("║      GPT-4o-mini via OpenRouter          ║")
    print("╚══════════════════════════════════════════╝")
    print("\nExamples:")
    print("  • Find me 3 internships in startups in India")
    print("  • Remote data science jobs posted today")
    print("  • Entry-level ML engineer roles in Bangalore\n")
    print("Type 'exit' or 'quit' to stop.\n")

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
        print("\n" + "─" * 50 + "\n")


if __name__ == "__main__":
    main()