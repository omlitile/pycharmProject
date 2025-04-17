import os
import env
from langchain_community.tools.tavily_search import TavilySearchResults
apiKey = os.getenv("TAVILY_API_KEY")
print(apiKey)

search = TavilySearchResults(max_result=2,tavily_api_key=apiKey)

print(search.invoke("今天佛山天气怎么样"))