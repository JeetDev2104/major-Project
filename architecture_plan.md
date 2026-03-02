# Autonomous Job Application AI System

## Phase 1: Planning & Agentic Architecture

### 1. The Core Architecture (LangGraph + Supervisor Model)
From **Repository 1 (`GENAI-CareerAssistant-Multiagent`)**, we will adopt the **Supervisor Multi-Agent Architecture** built on **LangGraph**. 
- **Why this works:** LangGraph is specifically designed for complex, stateful multi-agent workflows. It allows us to define a "Supervisor" node that can orchestrate clearly defined "worker" nodes (our council of AI agents) and manage the state (the job listings, the resume score, the emails) between them.
- **The Council of Agents:** The Supervisor will consult the council before allowing the Auto-Apply agent to execute.

### 2. The Power Tools (JobSpy & Playwright/Claude Code)
From **Repository 2 (`ApplyPilot`)**, we will extract their extremely powerful scraping and automation tools.
- **JobSpy:** We will use `python-jobspy` for the **Job Search Agent**. It is already optimized to bypass anti-bot measures and scrape major platforms (LinkedIn, Indeed, Glassdoor) natively.
- **Playwright & Claude Code:** We will use these for the **Auto-Apply Agent**. Using an LLM ( Claude Code ) connected to a headless browser (Playwright) via MCP allows for actual dynamic form filling and button clicking.

### 3. The Communication Layer
- **WhatsApp/Telegram:** We will build a custom endpoint (likely using Python + a WhatsApp/Telegram API webhook) that acts as the user interface. The Supervisor will ping this webhook to send updates to your phone ("Found 10 jobs, applying to 3...") and await your approval if required.

---

### The Council of Agents Workflow (Phase 1 focus)

1. **Job Search Agent (The Scout):** 
   - Uses `JobSpy` (from Repo 2).
   - Scrapes job platforms.
   - Outputs a structured CSV/Text file: `Company Name`, `Job Link`, `Company Career Page`, `Source Platform`.
2. **Resume Matching Agent (The Analyst):**
   - Parses the user's resume (from Repo 1 tools).
   - Reads the JD scraped by the Scout.
   - Outputs an AI Fit Score (1-100) and tailored keywords.
3. **Cold Emailing Agent (The Networker):**
   - Uses web scrapers (like `FireCrawl` from Repo 1) to find recruiter emails on the company's career page or via LinkedIn.
   - Drafts and queues an autonomous cold email.
4. **Auto-Apply Agent (The Executor):**
   - Receives highly-scored jobs from the Supervisor.
   - Uses Playwright/Claude Code to navigate the `Job Link` and submit the application.
5. **Supervisor Agent (The Boss):**
   - Evaluates the Analyst's score. If score > 80, it tells the Executor to apply and the Networker to email.
   - Sends a summary message to the user via Telegram/WhatsApp.

---

## Next Step: Building the Job Search Agent
If you approve this architecture, we will begin writing the Python code for **Agent 1: The Job Search Agent**, which will utilize `JobSpy` to fetch job listings and save them directly to a CSV file!
