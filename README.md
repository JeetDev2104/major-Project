# 🧑‍💼 AI Job Search Agent

An intelligent job search agent powered by **GPT-4o-mini via OpenRouter**. Ask it in plain English — it searches LinkedIn, understands follow-up questions, and saves results to a CSV automatically.

---

## ✨ Features

- 🔍 **Natural language queries** — "Find me 3 Python internships in startups in India"
- 🧠 **Conversation memory** — ask follow-ups like "which of those pays the most?"
- 💼 **LinkedIn job search** — pulls real listings with apply links
- 🌐 **Google search** — used for salary/company research on follow-up questions
- 📄 **Auto CSV export** — every search saves full job details to `scouted_jobs.csv`
- 🇮🇳 **India-first defaults** — defaults to India if no location is mentioned

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/JeetDev2104/major-Project.git
cd major-Project
```

### 2. Install dependencies
```bash
pip install openai python-dotenv langchain-core langchain-community \
            linkedin-api aiohttp beautifulsoup4 pydantic firecrawl-py
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```env
OPEN_ROUTER_API_KEY=your_openrouter_api_key
LINKEDIN_EMAIL=your_linkedin_email
LINKEDIN_PASS=your_linkedin_password
SERPER_API_KEY=your_serper_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

> **Get your keys:**
> - OpenRouter (free): https://openrouter.ai
> - Serper (free tier): https://serper.dev
> - FireCrawl (free tier): https://firecrawl.dev

### 4. Run the agent
```bash
python3 Job_Agent.py
```

---

## 💬 Example Queries

```
👉  Find me 3 internships in startups in India
👉  Remote AIML engineer jobs in India
👉  Entry-level data scientist roles in Bangalore
👉  Which of those companies pays the most?
👉  Tell me more about job 2
👉  Python developer jobs posted in the last 24 hours
```

---

## 📁 Output — scouted_jobs.csv

Every LinkedIn search automatically saves results to `scouted_jobs.csv`:

| Search Query | Job Title | Company Name | Location | Job Type | Time Posted | Applicants | Job Description | Apply Link | Saved At |
|---|---|---|---|---|---|---|---|---|---|
| Python jobs India | Python Developer | Infosys | Bangalore | full-time | 1 week ago | — | As part of the Infosys... | linkedin.com/... | 2025-03-03 |

Open directly in **Excel**, **Google Sheets**, or **Apple Numbers**.

---

## 🗂️ Project Structure

```
major-Project/
├── Job_Agent.py      # Main agent — run this
├── tools.py          # Tool definitions (LinkedIn, Google, Scraper)
├── search.py         # LinkedIn job scraping logic
├── utils.py          # Serper & FireCrawl clients
├── CLI.py            # Reference CLI agent (agentic loop pattern)
├── config.yaml       # Config settings
├── scouted_jobs.csv  # Auto-generated output (gitignored)
└── .env              # API keys (never commit this!)
```

---

## ⌨️ Special Commands

| Command | Action |
|---|---|
| `reset` | Clear conversation memory and start fresh |
| `exit` / `quit` | Quit the agent |

---

## 🔒 Security Note

`.env` is gitignored. **Never commit your API keys or LinkedIn credentials.** Use `.env.example` as a template if sharing with others.
