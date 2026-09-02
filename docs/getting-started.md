# Getting Started — Read Me First

> **Start here.** This guide assumes you know how to use a computer (browse the web,
> make folders) but have **never** written code before. By the end of this page you
> will have the course running on your computer and have run your very first code cell.

This course has ~200 hands-on notebooks that take you from "hello world" in Python all
the way to building your own AI assistant. You are starting at the perfect time — the
curriculum is designed so that **everything is taught when you need it**.

---

## 1. The Journey in Plain Words

```text
You                  →  Build your own AI system
 │
 ├─ Week 1-2   Environment + Python (Phases 00-01)
 │             "I can write programs: variables, loops, functions"
 │
 ├─ Month 2    Math + Statistics (Phases 02-03)
 │             "I understand why algorithms work, not just how"
 │
 ├─ Month 3    Data + Machine Learning (Phases 04-05)
 │             "I can clean data and train real models"
 │
 ├─ Month 4-5  Deep Learning + NLP (Phases 06-08)
 │             "I can build neural networks and work with text"
 │
 ├─ Month 6-7  Generative AI + RAG (Phases 09-13)
 │             "I can build apps that talk to LLMs"
 │
 ├─ Month 8    Agents + Evaluation (Phases 14-15)
 │             "I can build AI that uses tools and I can prove it works"
 │
 └─ Month 9    Deployment + Capstone (Phases 16-17)
               "I ship a real AI product for my portfolio"
```

> These are **beginner-paced estimates, full-time**. Go slower or faster — depth beats speed.
> Each phase has a real project; the last one (Phase 17) is a portfolio capstone you design yourself.

---

## 2. Glossary First? Yes — 5 Terms Only

You'll meet these words in the next 10 minutes. The full list is in [`docs/glossary.md`](glossary.md).

| Word | Plain meaning |
|------|---------------|
| **Terminal / PowerShell / Command Line** | A window where you type commands instead of clicking. (Finder on Mac is called *Terminal*.) |
| **Python** | The programming language this whole course uses. It must be installed first. |
| **Virtual environment (`.venv`)** | A private folder inside this project holding Python + the libraries. Keeps your computer clean. |
| **Package / library** | Ready-made code others wrote (like NumPy for math). You install them when the course tells you. |
| **Jupyter notebook (`.ipynb`)** | A document mixing text cells and code cells. You'll live in these for the whole course. |

---

## 3. What You Need

- **A laptop/desktop** (Windows, Mac, or Linux). A phone won't work.
- **Internet** for downloading Python and libraries.
- **About 10–20 GB free disk space** (AI libraries are chunky).
- A GPU is **NOT required** — everything runs on a normal CPU, just slower.

You do **NOT** need: a paid API key, a degree, or any prior programming experience.

---

## 4. First Session — Step by Step (20–30 minutes)

> **If you're on Windows**, "terminal" = **PowerShell** (search it in the Start menu).
> On **Mac**, it's **Terminal**. On **Linux**, also Terminal.

### Step 1: Install Python (3.13 or newer)

1. Go to <https://www.python.org/downloads/>
2. Download the latest Python 3.13.x installer for your system.
3. **Important (Windows):** tick the box **"Add Python to PATH"** during install.
4. Mac users: install with the installer, or `brew install python@3.13`.

### Step 2: Install Git

For downloading the course and tracking your changes.

- **Windows:** <https://git-scm.com/download/win> (install with defaults)
- **Mac:** `brew install git`
- **Linux:** `sudo apt install git` (Ubuntu/Debian)

### Step 3: Install uv (the tool that installs the libraries)

Open your **terminal** and paste:

```powershell
pip install uv
```

Follow the instructions → you'll see it uses `uv` like you'd use WhatsApp to install apps, but for Python libraries.

### Step 4: Download this course

In your terminal, go to the folder where you keep projects, then:

```bash
git clone https://github.com/5at4am/complete-data-science.git
cd complete-data-science
```

`cd` means **"change directory"** — you're telling the terminal to step inside the course folder.

### Step 5: Install all libraries (one command)

```bash
uv sync
```

This downloads every library the course uses. **First time can take 5–15 minutes** — that's normal.
It creates the `.venv` folder (your private toolbox).

### Step 6: Activate the environment

```bash
# Windows (PowerShell)
.\.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

You'll know it worked because `(.venv)` appears at the start of your prompt.

### Step 7: Launch Jupyter

```bash
jupyter lab
```

Your browser opens with a file browser. Navigate into **`notebooks/00_environment/`** and open
**`00_environment_verification.ipynb`** — a notebook that proves everything works.

### Step 8: Run your first code cell 🎉

Inside the notebook, click the first code cell, then press **Shift + Enter**. That's the magic shortcut
for "run this cell". The notebook will check Python, all libraries, and internet — every check turning
green is evidence your setup works.

> **Your win condition for today:** the verification notebook finishes with no red errors.
> If you see errors, go to [`docs/troubleshooting.md`](troubleshooting.md) — it's written for you.

---

## 5. If Something Goes Wrong (First-Hour Rescue)

| Symptom | What it means | Fix |
|---|---|---|
| `'git' is not recognized...` | Git not installed or not on PATH | Install Git, restart the terminal |
| `'uv' is not recognized...` | uv not installed or PATH issue | `pip install uv`, then **close & reopen** the terminal |
| `uv sync` very slow / errors | Network or disk | Check internet; free 10+ GB disk; retry |
| `jupyter` not recognized | Environment not activated | Back to Step 6 — your prompt should show `(.venv)` |
| Notebook opens but code cells show `ModuleNotFoundError` | Wrong Python kernel | See "Jupyter kernel doesn't see packages" in [`docs/troubleshooting.md`](troubleshooting.md) |
| Page that opens looks empty | Wrong folder | In Jupyter, the sidebar is a file browser — navigate to `notebooks/00_environment/` |

---

## 6. Your First Week (Recommended Rhythm)

- **Day 1:** Finish the First Session above (Steps 1–8).
- **Days 2–3:** Phase 00 (notebooks `00.1` → `00.4`) — environment, Jupyter, Git, packaging.
- **Days 4–7:** Start Phase 01 (Python) — finish `01.1` to `01.4`. Do the **Hands-On Practice** cells —
  they are graded by difficulty (Basic → Challenge), and that's where the real learning happens.

**Daily loop (works for every notebook):**
1. Read the "What Are We Solving?" section.
2. Run cells in order. **Type, don't copy-paste** — muscles remember.
3. Do the Hands-On Practice with your own attempts first.
4. Answer the Knowledge Check out loud (to a rubber duck if no human is around).
5. Tick the **Exit Criteria** boxes. Only then move on.

---

## 7. You'll Also See These Files — Ignore Them For Now

| File | Purpose | When to care |
|---|---|---|
| `tracking/progress.md` | A checklist of every unit | From Day 2 — tick boxes as you finish |
| `tracking/mistakes.md` | A notebook for your errors ("best mistakes to make + why") | As soon as you hit your first bug |
| `roadmap/ROADMAP.md` | The full plan | When you want to peek ahead |
| `docs/*.md` | Reference guides (math, stats, ML...) | When you forget something |
| `src/`, `tests/`, `projects/` | Reusable code, tests, and hands-on projects | From Phase 05 onwards |

---

## 8. Habits That Separate Doers From Tutorial-Zombies

1. **Always restart the kernel before claiming a notebook works.** Kernel → Restart & Run All.
2. **Type the code.** Copy-paste gives you *recognition*, typing gives you *memory*.
3. **Break it on purpose.** Change a number, delete a line — observe the error, read it, fix it.
4. **Write the error you fixed into `tracking/mistakes.md`.** It becomes private review material.
5. **Explain to a rubber duck.** If you can't explain a cell, you haven't learned it yet.
6. **Never skip the hands-on cells.** They are the curriculum; the reading is scaffolding.

---

## 9. Mini FAQ

**Q: I've never seen code. Should I read Python docs first?**
No. Phase 01 teaches everything you need, in small units, with the environment you just set up.

**Q: How long is the whole course really?**
Full-time: ~9 months to a completed portfolio capstone. Part-time (2h/day): ~2 years.
It's a marathon by design — depth is the point.

**Q: Do I need Windows/Mac knowledge?**
Just the basics — making folders, double-clicking installers, copy-paste.

**Q: My math is rusty. Is that fatal?**
No. Phases 02–03 rebuild what you need *exactly when you need it*, implementation-first.

**Q: Do I need money / a credit card?**
No. Everything runs locally. (Only Phases 09+ use optional LLM APIs, and local alternatives are provided.)

**Q: I'm stuck past the first hour.**
Read the error message → search the exact error text online → check
[`docs/troubleshooting.md`](troubleshooting.md) → record the mistake in `tracking/mistakes.md` → keep going.

---

**Next step:** read [`docs/glossary.md`](glossary.md) lightly (skim once, don't memorize), then
go back to the README and begin **Phase 00**. You are set up, and the course will do the rest.
See you at the capstone.