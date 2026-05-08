import requests

class SearXNGEngine:
    def __init__(self, url="http://localhost:8080"):
        self.url = url
    def search(self, query, max_results=5):
        resp = requests.get(f"{self.url}/search", params={"q": query, "format": "json"})
        return [{"title": r["title"], "url": r["url"], "snippet": r.get("content","")} for r in resp.json().get("results", [])[:max_results]]
