import time
import json
from pathlib import Path
import re
from collections import defaultdict
import math
import os
from groq import Groq


DATA_PATH = Path("data//companies.jsonl")

def load_companies(path):
        #Read all companies from the JSONL file and return them as a list of dictionarieso
    companies = []

    with open (path ,"r", encoding="utf-8") as file:
      for line in file:
           line = line.strip()   # remove whitespace and newlines
           if line:              # skip empty lines
                company = json.loads(line)   # convert JSON string to dictionary
                companies.append(company)       
    print(f"Loaded {len(companies)} companies")
    return companies

#Extract hard numeric and boolean constraints from the query
#Returns a dictionary with the active constraints found
def parse_structured_constraints(query) :
    constraints = {}
    q = query.lower()

    #employee count
    m = re.search(r"more than\s+([0-9,]+)\s+employees?",q)
    if m:
        constraints["min_employees"] = int(m.group(1).replace(",",""))

    m = re.search(r"fewer than\s+([0-9,]+)\s+employees?", q)
    if m:
        constraints["max_employees"] = int(m.group(1).replace(",", ""))

    m = re.search(r"under\s+([0-9,]+)\s+employees?", q)
    if m:
        constraints["max_employees"] = int(m.group(1).replace(",", ""))

    #revenue
    m = re.search(r"revenue over\s+\$?([0-9,]+)\s*(million|m\b)?", q)
    if m:
        value = int(m.group(1).replace(",", ""))
        if m.group(2):  # if "million" or "m" was mentioned
            value = value * 1_000_000
        constraints["min_revenue"] = value

    #founded year
    m = re.search(r"founded after\s+(\d{4})", q)
    if m:
        constraints["min_founded"] = int(m.group(1))

    m = re.search(r"founded before\s+(\d{4})", q)
    if m: 
        constraints["max_founded"] = int(m.group(1))
    
    #public/private
    if re.search(r"\bpublic\b", q):
        constraints["is_public"] = True

    return constraints

#Remove companies that fail the hard constraints
#Companies with missing data are kept
def apply_structured_filter(companies, constraints):
    
    #if no constraints were found in the query return all the companies
    if not constraints:
        return companies

    qualified = []

    for company in companies:
        #check minimum employees
        if "min_employees" in constraints:
            employee_count = company.get("employee_count")
            if employee_count is None:
                pass # missing data -> keep the company
            elif employee_count <=constraints["min_employees"]:
                continue

        #check maximum employees
        if "max_employees" in constraints:
            employee_count = company.get("employee_count")
            if employee_count is None:
                pass  # missing data - keep
            elif employee_count >= constraints["max_employees"]:
                continue
        
        #check minimum revenue
        if "min_revenue" in constraints:
            revenue = company.get("revenue")
            if revenue is None:
                pass  # missing data - keep
            elif revenue < constraints["min_revenue"]:
                continue
        
        #check founded year
        if "min_founded" in constraints:
            year = company.get("year_founded")
            if year is None:
                pass  # missing data - keep
            elif year <= constraints["min_founded"]:
                continue 

        if "max_founded" in constraints:
            year = company.get("year_founded")
            if year is None:
                pass  # missing data - keep
            elif year >= constraints["max_founded"]:
                continue            
        
        #check public status
        if "is_public" in constraints:
            if company.get("is_public") != constraints["is_public"]:
                continue
        
        # company passed all checks
        qualified.append(company)

    print(f"Structured filter: {len(qualified)}/{len(companies)} companies passed")
    return qualified

#Flatten all relevant fields of a company into a single string
def company_to_text(company):
    parts = []

    simple_fields =[
        "operational_name",
        "address",
        "descrription",
    ]
    for field in simple_fields:
        value =company.get(field)
        if value is not None:
            parts.append(str(value))
    
    list_fields = [
        "core_offerings",
        "target_markets",
        "business_model",
    ]
    for field in list_fields:
        value =company.get(field)
        if value is not None and isinstance(value,list):
            parts.append(" ".join(str(item) for item in value))

    #primary naics is a dictionary from where the label will be extracted
    primary_naics = company.get("primary_naics")
    if primary_naics is not None and isinstance(primary_naics, dict):
        parts.append(primary_naics.get("label", ""))
    
    #secondary naics is a list of dictionaries
    secondary_naics = company.get("secondary_naics")
    if secondary_naics is not None and isinstance(secondary_naics, list):
        for naics_item in secondary_naics:
            if isinstance(naics_item, dict):
                parts.append(naics_item.get("label", ""))

    full_text = " ".join(parts)
    return full_text.lower()

#split a text into individual words
def tokenize(text):
    tokens = re.findall(r"[a-z]+", text)
    return tokens

#Build the TF-IDF data structures for all companies.
#   Returns:
#      - tokenised_docs: a list where each item is the list of words for that company
#      - idf: a dictionary mapping each word to its IDF score
def build_tfidf_index(companies):
    
    tokenised_docs= []

    #company -> text -> list of words -> add to the big list(tokenised_docs)
    for company in companies:
        text = company_to_text(company)
        tokens= tokenize(text)
        tokenised_docs.append(tokens)
    
    total_companies = len(tokenised_docs)

    #count in how many companies each word appears
    document_frequency = defaultdict(int)

    for tokens in tokenised_docs:
        unique_tokens = set(tokens) #remove duplicates
        for token in unique_tokens:
            document_frequency[token]+=1

    idf= {}

    for token, doc_count in document_frequency.items():
        idf[token] =math.log((total_companies+1)/(doc_count+1)) +1

    return tokenised_docs, idf

#Score all companies against the query using TF-IDF and returns top k most relevant companies
def tfidf_top_k(query, companies, k):
    
    if not companies:
        return []
    
    tokenised_docs, idf = build_tfidf_index(companies)
    query_tokens = tokenize(query.lower())

    scores= []

    for index, tokens in enumerate(tokenised_docs):
        #count how many times each word appears in this company's text
        word_count = defaultdict(int)
        for token in tokens:
            word_count[token] +=1
        total_words = len(tokens)
        if total_words == 0:
            scores.append(0.0)
            continue
        # calculate the TF-IDF score for this company
        company_score = 0.0
        for query_token in query_tokens:
            tf = word_count[query_token] / total_words
            token_idf = idf.get(query_token, 0)
            company_score += tf * token_idf

        scores.append(company_score)
             
    # pair each company with its score and sort highest first
    companies_with_score = list(zip(scores, companies))
    def get_score(pair):
        return pair[0]
    companies_with_score.sort(key=get_score, reverse=True)

    top_k_companies = []
    for score, company in companies_with_score[:k]:
        top_k_companies.append(company)
    
    print(f"TF-IDF: selected top {len(top_k_companies)} candidates")
    return top_k_companies

#Build a compact text summary of a company for the LLM prompt
def build_company_summary(company):
    lines = []
    if company.get("operational_name"):
        lines.append(f"Name: {company['operational_name']}")
    if company.get("address"):
        lines.append(f"Location: {company['address']}")
    if company.get("description"):
        description = company["description"][:300]
        lines.append(f"Description: {description}")
    if company.get("core_offerings"):
        offerings = company["core_offerings"]
        if isinstance(offerings, list):
            offerings_text= ", ".join(offerings)
        else:
            offerings_text = str(offerings)
        lines.append(f"Core Offerings: {offerings_text}")
    if company.get("target_markets"):
        markets = company["target_markets"]
        if isinstance(markets, list):
            markets_text= ", ".join(markets)
        else:
            markets_text = str(markets)
        lines.append(f"Target Markets: {markets_text}")
    if company.get("primary_naics"):
        naics = company["primary_naics"]
        if isinstance(naics, dict):
            lines.append(f"Industry: {naics.get('label', '')}")
    if company.get("employee_count"):
        lines.append(f"Employees: {company["employee_count"]}")
    if company.get("revenue"):
        lines.append(f"Revenue: {company['revenue']:,.0f}")
    if company.get("year_founded"):
        lines.append(f"Founded: {int(company['year_founded'])}")
    if company.get("is_public") is not None:
        lines.append(f"Public: {company['is_public']}")

    return "\n".join(lines)

# Send a batch of companies to Groq for qualification
def llm_qualify_batch(client, query, companies):

    #build the numbered list of companies for the prompt
    numbered_companies = ""
    for i, company in enumerate(companies):
        summary = build_company_summary(company)
        numbered_companies += f"[{i+1}]\n{summary}\n"
    
    #build the full prompt
    prompt = f"""You are a company qualification assistant. Evaluate whether each company truly satisfies the following user query.

USER QUERY: "{query}"

COMPANIES TO EVALUATE:
{numbered_companies}

For each company, respond with ONLY a JSON array where each element has:
- "id": the number shown in brackets (integer)
- "score": a float from 0.0 to 1.0 indicating how well this company matches the query
  (1.0 = perfect match, 0.0 = completely irrelevant, 0.5 = borderline)
- "reason": one concise sentence explaining your score

Be strict: only companies that genuinely satisfy the query intent should score above 0.6.
Respond with the JSON array only. No extra text, no markdown."""

    #make the API call
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        max_tokens = 2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    #extract the text from the response
    raw_text= response.choices[0].message.content.strip()


    # clean up any markdown formatting the model might have added
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*", "", raw_text)
    raw_text = raw_text.strip()

    #parse the JSON response
    try:
        results = json.loads(raw_text)
    # if that fails, extract data with regex as fallback
    # this works even when reason values have no quotes
    except json.JSONDecodeError:
      
        results = []
        pattern = r'"id"\s*:\s*(\d+).*?"score"\s*:\s*([0-9.]+).*?"reason"\s*:\s*"?([^"}\n][^}\n]*?)"?\s*[,}]'
        matches = re.findall(pattern, raw_text, re.DOTALL)
        for match in matches:
            results.append({
                "id": int(match[0]),
                "score": float(match[1]),
                "reason": match[2].strip()
            })

        if not results:
            print(f"  Warning: could not parse LLM response, skipping batch")
            print(f"  Problematic response: {raw_text[:300]}")
            return []

    #match each result back to its company
    scored_companies = []
    for result in results:
        company_id=result.get("id")
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")

        #company id starts at 1 but index starts at 0
        company_index = company_id -1

        if 0<= company_index< len(companies):
            company = companies[company_index]
            scored_companies.append((company, score, reason))
    
    return scored_companies

#Run LLM qualofication over all candidates in batches of 10
#returns all resulsts sorted by score descending
def llm_qualify_all(client, query, candidates):
    BATCH_SIZE = 10
    all_results = []
    total_batches = (len(candidates)+BATCH_SIZE-1)//BATCH_SIZE

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i: i+BATCH_SIZE]
        batch_number = (i//BATCH_SIZE) + 1

        print(f" LLM batch {batch_number}/{total_batches} ({len(batch)} companies)...")

        results= llm_qualify_batch(client, query, batch)
        all_results.extend(results)

        time.sleep(0.3)

    def get_score(result_tuple):
        return result_tuple[1]
    
    all_results.sort(key=get_score, reverse= True)
    return all_results

# Run this to verify everything works
if __name__ == "__main__":
    companies = load_companies(DATA_PATH)
    client = Groq()

    test_queries = [
       # "Public software companies with more than 1,000 employees",
       # "Clean energy startups founded after 2018 with fewer than 200 employees",
        #"Construction companies in the United States with revenue over $50 million",
        "Logistics companies in Romania",
       # "Pharmaceutical companies in Switzerland",
    ]

    for query in test_queries:
       # print(f"\nQUery: {query}")
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")

        constraints = parse_structured_constraints(query)
        #print(f"Constraints found: {constraints}")
        result = apply_structured_filter(companies, constraints)
        #print(f"Companies remaining: {len(result)}")
        top_30 = tfidf_top_k(query, result, k=30)
    
        print("Running LLM qualification...")
        results = llm_qualify_all(client, query, top_30)

        print(f"\nTop 5 candidates:")
        for i, (company, score, reason) in enumerate(results[:5], 1):
            name = company.get("operational_name", "Unknown")
            print(f"  {i}. [{score:.2f}] {name}")
            print(f"   {reason}")



    
    