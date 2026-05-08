from duckduckgo_search import DDGS

class DuckDuckGoAPI:
    def search(self, query, max_results=5):
        with DDGS() as ddgs:
            return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in ddgs.text(query, max_results=max_results)]
