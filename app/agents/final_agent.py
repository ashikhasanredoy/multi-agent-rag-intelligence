from typing import Dict, Any, List
from app.models.schemas import ChatResponse, SourceCitation
from app.utils.logging import logger


class FinalAgent:
    """Final Response Synthesizer & Formatter."""

    def _filter_relevant_sources(self, draft_answer: str, query: str, sources: List[SourceCitation]) -> List[SourceCitation]:
        """Keep only verified sources that were actually used / retrieved for the answer."""
        if not sources:
            return []

        lower_answer = draft_answer.lower()
        lower_query = query.lower()

        # If answer states that no information / no evidence was found to answer the query,
        # do NOT show useless random search links
        no_info_markers = [
            "does not contain any information regarding",
            "does not contain information",
            "no available answer",
            "none of which address",
            "[no_relevant_docs]",
            "no matching documents found",
            "could not find any information"
        ]
        if any(marker in lower_answer for marker in no_info_markers) and not any(k in lower_answer for k in ["policy specifies", "policy states", "according to"]):
            return []

        # Standard stop words to ignore when checking match
        stop_words = {
            "the", "and", "for", "with", "from", "this", "that", "official",
            "summary", "guide", "release", "complete", "examples", "platform",
            "about", "wikipedia", "research", "evidence", "document", "online",
            "mobile", "overview", "what", "which", "news", "articles", "http",
            "https", "yahoo", "finance", "data", "market"
        }

        import re
        relevant = []
        seen_urls = set()

        for s in sources:
            url = (s.url or "").strip()
            doc_name = (s.document or "").strip()
            snippet = (s.snippet or "").strip()

            if url and url in seen_urls:
                continue

            # 1. Internal Knowledge Base Documents (e.g. PDFs, TXT policy files)
            if any(doc_name.lower().endswith(ext) for ext in [".pdf", ".txt", ".md", ".docx"]):
                # Keep if document name or substantial terms are in the answer
                base_name = doc_name.rsplit(".", 1)[0].lower()
                doc_terms = [w for w in re.findall(r'[a-zA-Z0-9]{3,}', base_name) if w not in stop_words]
                if any(t in lower_answer for t in doc_terms) or not doc_terms:
                    relevant.append(s)
                    if url:
                        seen_urls.add(url)
                continue

            # 2. Finance quotes (symbols like AAPL, MSFT, etc.)
            if getattr(s, "source_type", None) == "finance" or "yahoo finance" in doc_name.lower():
                # Check ticker or company in answer
                symbol_match = re.search(r'\b([A-Z0-9\.\^]{1,6})\b', doc_name)
                if symbol_match and symbol_match.group(1).lower() in lower_answer:
                    relevant.append(s)
                    if url:
                        seen_urls.add(url)
                    continue

            # 3. Web and Wikipedia sources
            # Title words check: title must be discussed in the answer
            title_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc_name.lower())
            title_words = [w for w in title_clean.split() if len(w) > 2 and w not in stop_words]

            if title_words:
                matched_words = [w for w in title_words if w in lower_answer]
                # Require majority of title terms to be actively mentioned in the generated answer
                if len(matched_words) >= max(1, (len(title_words) + 1) // 2):
                    relevant.append(s)
                    if url:
                        seen_urls.add(url)
                    continue

            # Snippet overlap check (at least 3 content words from snippet in answer)
            if snippet:
                snip_words = [w for w in re.findall(r'[a-zA-Z]{4,}', snippet.lower()) if w not in stop_words]
                shared_words = [w for w in snip_words if w in lower_answer]
                if len(shared_words) >= 3:
                    relevant.append(s)
                    if url:
                        seen_urls.add(url)
                    continue

        return relevant

    async def format_response(
        self,
        conversation_id: str,
        query: str,
        draft_answer: str,
        agent_name: str,
        sources: List[SourceCitation],
        confidence: float,
        latency: float
    ) -> ChatResponse:
        logger.info(f"FinalAgent assembling response from {agent_name}")
        
        # Filter sources to ensure only relevant citations retrieved for this answer are kept
        filtered_sources = self._filter_relevant_sources(draft_answer, query, sources)
        
        return ChatResponse(
            conversation_id=conversation_id,
            answer=draft_answer,
            agent=agent_name,
            sources=filtered_sources,
            confidence=confidence,
            latency_seconds=latency
        )


final_agent = FinalAgent()
