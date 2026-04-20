from exa_py import Exa
import os
from dotenv import load_dotenv

load_dotenv()

exa = Exa(api_key=os.getenv("EXA_API_KEY"))

result = exa.search("Ranjan Shettigar", category="people", type="auto", num_results=1)
print(result)
