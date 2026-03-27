import json
from pathlib import Path

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