import json
from pathlib import Path
import re
from collections import defaultdict
import math

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
        value = company.get(field)
        if value is not None:
            parts.append(str(value))



# Run this to verify everything works
if __name__ == "__main__":
    companies = load_companies(DATA_PATH)

    # Show the first company so you can see what the data looks like
    print("\nFirst company in the dataset:")
    print(json.dumps(companies[0], indent=2))

    # Show some basic statistics
    print(f"\nCompanies with a description: {sum(1 for c in companies if c.get('description'))}")
    print(f"Companies with an address: {sum(1 for c in companies if c.get('address'))}")
    print(f"Public companies: {sum(1 for c in companies if c.get('is_public') == True)}")

    print(f"test part2")
    test_queries = [
        "Public software companies with more than 1,000 employees",
        "Clean energy startups founded after 2018 with fewer than 200 employees",
        "Construction companies in the United States with revenue over $50 million",
    ]

    for query in test_queries:
        print(f"\nQUery: {query}")
        constraints = parse_structured_constraints(query)
        print(f"Constraints found: {constraints}")
        result = apply_structured_filter(companies, constraints)
        print(f"Companies remaining: {len(result)}")