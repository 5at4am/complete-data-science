# Knowledge Base — Provenance & Policy

## What goes here

This is the **only** source of information the TaxKraft Support Assistant is allowed to
answer from. If it is not in this directory (or in `crawled/` after a crawl), the chatbot
must *not* answer it — the topic-scope guardrail enforces exactly that.

## Policy (the "company-only" rule)

1. Every document starts with an HTML comment **provenance block** (see below).
2. Facts are written conservatively — service names, categories, contact details, and the
   company narrative that TaxKraft publishes itself.
3. Specific, time-sensitive claims (prices, timeline days) that we did not independently
   verify on the live website **must be flagged** with `VERIFY:` so the crawler refresh
   either confirms or replaces them before the system goes live.
4. No competitor information, no speculation about other companies, no inferred opinions.

## Provenance block format

```markdown
<!--
source-title: <Human readable page name>
source-url: <public URL>
verified-at: <YYYY-MM-DD>
provenance: public website (sitemap + schema.org JSON-LD + public marketing copy)
-->
```

## Refresh workflow

```powershell
python -m crawler.sitemap          # lists current sitemap URLs -> crawler/raw/sitemap.json
python -m crawler.fetch            # fetches pages (HTML + meta). Note: taxkraft.com is an SPA,
                                   # so run the optional Playwright renderer for full text.
python -m crawler.build_corpus     # writes knowledge_base/crawled/*.md with provenance
python run.py ingest               # re-embeds everything
```

The seed corpus below is dated **2026-08-29** and sourced from the public sitemap
(`https://taxkraft.com/sitemap.xml`), the homepage/about/contact pages, and TaxKraft's
public schema.org JSON-LD and LinkedIn company page.

## Document index

| File | Topic(s) |
|---|---|
| `company_overview.md` | Who TaxKraft is, mission, address, contact |
| `services_company_registration.md` | Pvt Ltd, Public Ltd, LLP, OPC, Partnership, Section 8, Sole Proprietorship, Startup India |
| `services_gst.md` | GST registration, monthly/quarterly/annual returns, e-commerce, cancellation, notices, LUT |
| `services_income_tax.md` | ITR-1..7, TDS/TCS returns, assessments, income tax notices, 12A/80G, 80-IAC, FCRA, 15CA/15CB |
| `services_compliance.md` | Pvt Ltd / LLP / Partnership annual compliance, ROC filings |
| `services_registrations_licenses.md` | Trademark, MSME, Udyam, FSSAI, IEC, ISO, GEM, Shop & Est., PF, ESI, PAN, TAN, DSC, BIS, ISI, LEI |
| `services_financial_advisory.md` | CFO services, due diligence, project reports, loans (business/retail/term/working capital) |
| `pricing_and_process.md` | How TaxKraft works, consultation, reminders, budgets — VERIFY flagged |
| `faq_general.md` | Common support questions |
| `contact_support.md` | Phone, email, office, languages, partner program |