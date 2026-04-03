# WRITEUP — Company Qualification System

##Approach

### System Architecture

The system is structured as a three-stage pipeline that progressively narrows
down candidates from the full dataset to a final ranked list. Each stage is
designed to eliminate clearly irrelevant companies cheaply before the expensive
stage runs.

Full dataset (477 companies)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Structured Filter │  — free, instant, deterministic
│                             │    extracts hard constraints from query
│  employee count, revenue,   │    drops companies that provably fail
│  founded year, is_public    │
└──────────┬──────────────────┘
           │ filtered subset
           ▼
┌─────────────────────────────┐
│  Stage 2: TF-IDF Ranking    │  — fast, no API cost
│                             │    converts company profiles to text
│                             │    scores against query tokens
│                             │    selects top 30 candidates
└──────────┬──────────────────┘
           │ top 30 candidates
           ▼
┌─────────────────────────────┐
│  Stage 3: LLM Qualification │  — accurate, targeted
│  (Llama 3.3 70B via Groq)   │    groups candidates in batches of 10
│                             │    scores each company 0-1 with reason
└──────────┬──────────────────┘
           │ scored and filtered
           ▼
    Final ranked list

### Stage 1 — Structured Filter

Before any semantic reasoning, hard constraints are extracted directly from
the query using regex patterns. This stage is free, deterministic, and
eliminates companies that probably fail numerical or boolean constraints.

Constraints extracted include: employee count bounds, revenue thresholds,
founding year, and public/private status.

Companies with missing data for a given constraint are kept — we give them
the benefit of the doubt rather than eliminating them unfairly.

### Stage 2 — TF-IDF Pre-Ranking

Rather than sending all remaining companies to an LLM, a TF-IDF similarity
score is used to rank candidates by textual relevance to the query. This stage
runs entirely in memory with no external dependencies and completes in
milliseconds for hundreds of companies.

Each company is converted to a flat text blob combining: name, address,
description, NAICS industry labels, core offerings, target markets, and
business model. Missing fields are skipped.

The top 30 candidates are passed to the LLM stage.

### Stage 3 — LLM Batch Qualification

Instead of one API call per company, candidates are sent in batches of 10 to
Llama 3.3 70B (via Groq) and asked to return a score (0-1) and one-sentence
reasoning for each company.

This design achieves:
- 10x cost reduction vs one-call-per-company
- Semantic reasoning for inferential queries
- Interpretability — every result has a human-readable reason
- Robustness — a fallback regex parser handles cases where the model
  returns malformed JSON

### Why This Design?

The pipeline matches cost to difficulty. Free structural filters run first,
then fast local TF-IDF computation, then LLM calls only on the 30 most
plausible candidates. By the time the LLM runs, most irrelevant companies
have already been eliminated without spending any API budget.


## Tradeoffs

### What I Optimised For

| Priority   | Choice 
|____________|_______________________________________________________________
| Cost       | LLM called on at most 30 companies per query, in batches of 10 
| Speed      | Stages 1 and 2 run in milliseconds 
| Accuracy   | LLM handles inferential reasoning that TF-IDF cannot 
| Simplicity | No vector database, no embeddings API, no infrastructure 

### Intentional Tradeoffs

**TF-IDF vs Embeddings**
I chose TF-IDF over embeddings to avoid an external embedding API dependency
and keep the system self-contained. The downside is that TF-IDF misses
semantic similarity when the query and company description use different
vocabulary. A company describing itself as a "lithium cell sub-assembly
manufacturer" might be missed by a query for "EV battery components".

**Small K (top 30)**
Passing only 30 candidates to the LLM is aggressive. A highly relevant company
ranked 35th in TF-IDF will be missed. I accept this tradeoff because
increasing K to 100 would triple the API cost. In practice, companies that
genuinely match a query tend to also use matching vocabulary in their
descriptions.

**Score threshold at 0.65**
Companies scoring below 0.65 are excluded from results. This removes borderline
matches like real estate companies that "touch" the logistics sector without
being logistics companies. The threshold can be adjusted per query type if
needed.

## Error Analysis

### Where the System Struggles

**Case 1: Inferential queries**

Query: *"Fast-growing fintech companies competing with traditional banks"*

TF-IDF surfaces companies containing "fintech" or "banking" vocabulary but
misses companies that compete with banks without explicitly saying so. The LLM
then qualifies from a pool that may not contain the best candidates.

Result: only 1 company qualified for this query, which is likely an
under-representation of the actual matches in the dataset.

**Case 2: Geography inference**

Some companies store their address as a dictionary with a country_code field
rather than a plain text string containing the country name. A company with
address `{'country_code': 'ro'}` may score lower for "companies in Romania"
than one that writes "Romania" explicitly in its description.

**Case 3: LLM output variability**

The same company can receive different scores across runs because the model
is not fully deterministic. A company scoring 0.90 in one run might score
0.60 in another, causing it to fall below the threshold and disappear from
results.

**Case 4: Duplicate results**

Some companies appear twice in the dataset with identical or near-identical
profiles. These duplicates can appear in the final results as if they were
two different companies.

## Scaling

### Handling 100,000+ Companies Per Query

The current pipeline is single-threaded and in-memory. At 100K companies,
several changes would be necessary:

**Pre-compute and persist the TF-IDF index**
Rebuild only when the dataset changes, not per query. Store the index on disk
and load it once at startup.

**Parallelise LLM calls**
Use async API calls to run multiple batches concurrently instead of
sequentially. This could reduce Stage 3 latency from ~30 seconds to ~5
seconds.

**Add a query router**
Detect purely structural queries (all constraints are numeric or boolean)
and return results after Stage 1 alone, with zero API cost.


## Failure Modes

### Confident but Wrong Results

**Mode 1: LLM inference from name alone**
If a company has a sparse profile (name and website only), the LLM scores
it based on the company name. A company called "GreenCell Technologies"
might score high for "EV battery components" even if it manufactures
agricultural sensors.

*Monitor by:* logging the percentage of high-scoring companies that had
fewer than 3 populated fields.

**Mode 2: TF-IDF vocabulary mismatch**
For queries using specialised jargon not present in company descriptions,
TF-IDF surfaces the wrong candidates. The LLM then qualifies from a pool
that doesn't include the right companies.

*Monitor by:* logging the distribution of TF-IDF scores for the top 30.
If all scores are below 0.01, the query vocabulary is absent from the
dataset.

**Mode 3: Structural filter over-elimination**
If a user writes "around 1000 employees" and the regex parses this as
min_employees=1000, borderline companies (950 employees) are eliminated
before the LLM can evaluate them.

*Monitor by:* alerting when Stage 1 eliminates more than 80% of the
dataset from a single constraint.
