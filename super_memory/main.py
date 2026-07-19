import os
from supermemory import Supermemory
from dotenv import load_dotenv

load_dotenv()

client = Supermemory(
    api_key=os.getenv("API_KEY"),  # Default, can be omitted
)

# Add a memory
client.add(content="Meeting notes from Q1 planning", container_tags=["user_123"])

# Search memories
response = client.search.documents(
    q="planning notes",
    container_tags=["user_123"]
)
print(response.results)

# Get user profile
profile = client.profile(container_tag="user_123")
print(profile.profile.static)
print(profile.profile.dynamic)
