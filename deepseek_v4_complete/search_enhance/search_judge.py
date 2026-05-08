class SearchJudge:
    def should_search(self, query):
        return any(kw in query for kw in ["最新","今天","实时","新闻"])
