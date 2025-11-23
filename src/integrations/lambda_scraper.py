"""Lambda Cloud-based web scraper for gathering recent jailbreak events and techniques."""

import asyncio
import httpx
import re
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.utils.logger import log
from src.integrations.lambda_cloud import LambdaCloudClient, LambdaModelRunner


@dataclass
class ScrapedEvent:
    """Represents a scraped event or technique."""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    category: str  # e.g., "jailbreak", "adversarial", "prompt_injection"
    relevance_score: float  # 0-1


class LambdaWebScraper:
    """
    Web scraper that uses Lambda Cloud instances for enhanced scraping and analysis.
    
    When instance_id is provided:
    - Runs parallel scraping locally (fast)
    - Uses Lambda instance API (if available) to enhance relevance scoring
    - Leverages cloud resources for better analysis
    
    Without instance_id:
    - Runs parallel local scraping (still fast)
    """
    
    def __init__(
        self,
        lambda_client: Optional[LambdaCloudClient] = None,
        instance_id: Optional[str] = None
    ):
        """
        Initialize Lambda-based web scraper.
        
        Args:
            lambda_client: Lambda Cloud client
            instance_id: Optional specific instance ID to use for enhanced analysis
        """
        self.lambda_client = lambda_client or LambdaCloudClient()
        self.instance_id = instance_id
        self.scraped_events: List[ScrapedEvent] = []
        if instance_id:
            log.info(f"LambdaWebScraper initialized with instance {instance_id} for enhanced scraping")
        
        # Search sources and queries
        self.search_queries = [
            "AI jailbreak techniques 2024",
            "LLM prompt injection attacks",
            "adversarial prompts for language models",
            "AI safety vulnerabilities",
            "GPT jailbreak methods",
            "Claude prompt engineering attacks",
            "recent AI model exploits"
        ]
        
        # Sources to scrape (using search engines and forums)
        self.sources = [
            "reddit.com/r/MachineLearning",
            "github.com",
            "arxiv.org",
            "twitter.com",
            "huggingface.co",
            "lesswrong.com",
            "ai.stackexchange.com"
        ]
    
    async def scrape_recent_events(
        self,
        days_back: int = 7,
        max_results: int = 50
    ) -> List[ScrapedEvent]:
        """
        Scrape recent jailbreak-related events and techniques.
        
        Args:
            days_back: How many days back to search
            max_results: Maximum number of results to return
            
        Returns:
            List of scraped events
        """
        log.info(f"Starting web scraping for recent events (last {days_back} days)")
        
        events = []
        
        # Use Lambda instance to run scraping tasks
        if self.instance_id:
            events = await self._scrape_with_lambda_instance(days_back, max_results)
        else:
            # Fallback: use local scraping (limited)
            events = await self._scrape_local(days_back, max_results)
        
        self.scraped_events.extend(events)
        log.info(f"Scraped {len(events)} recent events")
        
        return events
    
    async def _scrape_with_lambda_instance(
        self,
        days_back: int,
        max_results: int
    ) -> List[ScrapedEvent]:
        """Scrape using Lambda instance - runs scraping in parallel and uses API for enhanced analysis."""
        log.info(f"Using Lambda instance {self.instance_id} for enhanced scraping")
        
        # Get instance details
        instance = await self.lambda_client.get_instance_status(self.instance_id)
        if not instance:
            log.error(f"Instance {self.instance_id} not found, using local scraping")
            return await self._scrape_local(days_back, max_results)
        
        instance_ip = instance.get("ip")
        if not instance_ip:
            log.warning("Instance IP not available, using local scraping")
            return await self._scrape_local(days_back, max_results)
        
        # Run scraping tasks in parallel (fast)
        log.info(f"Running parallel scraping, will use Lambda API at {instance_ip} for analysis")
        
        # Use async parallel scraping
        tasks = [
            self._search_github(days_back),
            self._search_reddit(days_back),
            self._search_stackexchange(days_back),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        events = []
        for result in results:
            if isinstance(result, list):
                events.extend(result)
            elif isinstance(result, Exception):
                log.debug(f"Scraping task error: {result}")
        
        # Try to use Lambda instance API for enhanced analysis if available
        api_endpoint = f"http://{instance_ip}:8000/v1/chat/completions"
        enhanced_events = await self._enhance_with_lambda_api(events, api_endpoint, max_results)
        
        # Sort by relevance and limit
        enhanced_events.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        
        log.info(f"Scraped {len(enhanced_events)} events using Lambda instance")
        return enhanced_events[:max_results]
    
    async def _enhance_with_lambda_api(
        self,
        events: List[ScrapedEvent],
        api_endpoint: str,
        max_results: int
    ) -> List[ScrapedEvent]:
        """Use Lambda instance API to enhance event analysis and relevance scoring."""
        if not events:
            return events
        
        # Try to use Lambda API to analyze and score events
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test if API is available
                health_url = api_endpoint.replace("/v1/chat/completions", "/health")
                try:
                    health_response = await client.get(health_url, timeout=5.0)
                    if health_response.status_code != 200:
                        log.debug("Lambda API health check failed, skipping enhancement")
                        return events
                except:
                    log.debug("Lambda API not available, using events as-is")
                    return events
                
                # Use API to analyze top events for better relevance scoring
                for event in events[:min(10, len(events))]:  # Analyze top 10
                    try:
                        analysis_prompt = f"""Analyze this content for jailbreak/prompt injection relevance (0-1 score):
Title: {event.title[:200]}
Content: {event.content[:500]}

Respond with ONLY a number between 0 and 1 representing relevance score."""
                        
                        payload = {
                            "model": "gpt-3.5-turbo",  # Placeholder, vLLM will use loaded model
                            "messages": [{"role": "user", "content": analysis_prompt}],
                            "max_tokens": 10,
                            "temperature": 0.1
                        }
                        
                        response = await client.post(api_endpoint, json=payload, timeout=10.0)
                        if response.status_code == 200:
                            data = response.json()
                            if "choices" in data and len(data["choices"]) > 0:
                                score_text = data["choices"][0].get("message", {}).get("content", "").strip()
                                try:
                                    new_score = float(score_text)
                                    # Update relevance score (weighted average)
                                    event.relevance_score = (event.relevance_score * 0.5) + (new_score * 0.5)
                                except:
                                    pass
                    except Exception as e:
                        log.debug(f"Error enhancing event with Lambda API: {e}")
                        continue
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                
                log.info(f"Enhanced {min(10, len(events))} events using Lambda API")
        except Exception as e:
            log.debug(f"Lambda API enhancement failed: {e}, using events as-is")
        
        return events
    
    async def _scrape_local(
        self,
        days_back: int,
        max_results: int
    ) -> List[ScrapedEvent]:
        """
        Local scraping - runs all sources in parallel for speed.
        """
        log.info("Running local scraping (parallel)")
        
        # Run all scraping tasks in parallel for speed
        tasks = [
            self._search_github(days_back),
            self._search_reddit(days_back),
            self._search_stackexchange(days_back),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        events = []
        for result in results:
            if isinstance(result, list):
                events.extend(result)
            elif isinstance(result, Exception):
                log.debug(f"Scraping task error: {result}")
        
        # Sort by relevance and timestamp
        events.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        
        log.info(f"Local scraping found {len(events)} events")
        return events[:max_results]
    
    async def _search_duckduckgo(
        self,
        query: str,
        days_back: int
    ) -> List[ScrapedEvent]:
        """Search DuckDuckGo for recent results."""
        events = []
        
        try:
            # DuckDuckGo HTML search (simple scraping)
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    search_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                
                # Simple HTML parsing (in production, use BeautifulSoup)
                # For now, create synthetic results based on query
                if "jailbreak" in query.lower() or "prompt injection" in query.lower():
                    events.append(ScrapedEvent(
                        title=f"Recent {query} discussion",
                        content=f"Discussion about {query} techniques and methods",
                        source="DuckDuckGo Search",
                        url=search_url,
                        timestamp=datetime.now() - timedelta(days=1),
                        category="jailbreak",
                        relevance_score=0.8
                    ))
        except Exception as e:
            log.debug(f"DuckDuckGo search error: {e}")
        
        return events
    
    async def _search_github(self, days_back: int) -> List[ScrapedEvent]:
        """Search GitHub for recent repositories, issues, discussions, and code related to jailbreaks."""
        events = []
        
        try:
            # GitHub API (no auth needed for public repos, but rate limited to 60 requests/hour)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            search_terms = [
                "jailbreak",
                "prompt-injection",
                "adversarial-prompt",
                "llm-safety",
                "AI safety bypass",
                "GPT jailbreak"
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search repositories
                for term in search_terms[:4]:
                    try:
                        api_url = "https://api.github.com/search/repositories"
                        params = {
                            "q": f"{term} created:>{cutoff_str}",
                            "sort": "updated",
                            "order": "desc",
                            "per_page": 5
                        }
                        
                        response = await client.get(
                            api_url,
                            params=params,
                            headers={"Accept": "application/vnd.github.v3+json"}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            for item in data.get("items", [])[:3]:
                                # Get more details about the repo
                                repo_content = await self._get_github_repo_content(client, item.get("html_url", ""))
                                
                                events.append(ScrapedEvent(
                                    title=f"{item.get('name', '')} - {term}",
                                    content=f"{item.get('description', '')}\n\n{repo_content}",
                                    source="GitHub Repository",
                                    url=item.get("html_url", ""),
                                    timestamp=datetime.fromisoformat(
                                        item.get("updated_at", item.get("created_at", "")).replace("Z", "+00:00")
                                    ),
                                    category="jailbreak",
                                    relevance_score=0.8
                                ))
                        elif response.status_code == 403:
                            log.warning("GitHub API rate limit reached")
                            break
                        
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        log.debug(f"Error searching GitHub repos for {term}: {e}")
                
                # Search issues and discussions
                for term in search_terms[:2]:
                    try:
                        api_url = "https://api.github.com/search/issues"
                        params = {
                            "q": f"{term} is:issue created:>{cutoff_str}",
                            "sort": "updated",
                            "order": "desc",
                            "per_page": 5
                        }
                        
                        response = await client.get(
                            api_url,
                            params=params,
                            headers={"Accept": "application/vnd.github.v3+json"}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            for item in data.get("items", [])[:2]:
                                body = item.get("body", "")[:500]  # Limit body length
                                events.append(ScrapedEvent(
                                    title=item.get("title", ""),
                                    content=body,
                                    source="GitHub Issue",
                                    url=item.get("html_url", ""),
                                    timestamp=datetime.fromisoformat(
                                        item.get("updated_at", item.get("created_at", "")).replace("Z", "+00:00")
                                    ),
                                    category="jailbreak",
                                    relevance_score=0.75
                                ))
                        elif response.status_code == 403:
                            break
                        
                        await asyncio.sleep(1)
                    except Exception as e:
                        log.debug(f"Error searching GitHub issues for {term}: {e}")
                
                # Search code (look for actual jailbreak prompts in code)
                for term in ["jailbreak prompt", "prompt injection example"]:
                    try:
                        api_url = "https://api.github.com/search/code"
                        params = {
                            "q": f'"{term}" extension:py extension:md extension:txt created:>{cutoff_str}',
                            "sort": "indexed",
                            "order": "desc",
                            "per_page": 3
                        }
                        
                        response = await client.get(
                            api_url,
                            params=params,
                            headers={"Accept": "application/vnd.github.v3+json"}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            for item in data.get("items", [])[:2]:
                                code_url = item.get("html_url", "")
                                events.append(ScrapedEvent(
                                    title=f"Code: {item.get('name', '')}",
                                    content=f"Found in {item.get('repository', {}).get('full_name', '')}",
                                    source="GitHub Code",
                                    url=code_url,
                                    timestamp=datetime.now() - timedelta(days=1),
                                    category="jailbreak",
                                    relevance_score=0.9  # High relevance for code examples
                                ))
                        elif response.status_code == 403:
                            break
                        
                        await asyncio.sleep(1)
                    except Exception as e:
                        log.debug(f"Error searching GitHub code for {term}: {e}")
                        
        except Exception as e:
            log.debug(f"GitHub search error: {e}")
        
        return events
    
    async def _get_github_repo_content(self, client: httpx.AsyncClient, repo_url: str) -> str:
        """Get additional content from GitHub repository (README, etc.)."""
        try:
            # Convert HTML URL to API URL
            # e.g., https://github.com/user/repo -> https://api.github.com/repos/user/repo/readme
            match = re.search(r"github.com/([^/]+)/([^/]+)", repo_url)
            if match:
                owner, repo = match.groups()
                api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                
                response = await client.get(
                    api_url,
                    headers={"Accept": "application/vnd.github.v3+json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    import base64
                    content = base64.b64decode(data.get("content", "")).decode("utf-8")
                    return content[:1000]  # Limit length
        except Exception as e:
            log.debug(f"Error getting repo content: {e}")
        
        return ""
    
    async def _search_reddit(self, days_back: int) -> List[ScrapedEvent]:
        """Search Reddit for jailbreak discussions using Reddit API."""
        events = []
        
        try:
            # Reddit JSON API (no auth needed for public access)
            subreddits = [
                "r/MachineLearning",
                "r/artificial",
                "r/LocalLLaMA",
                "r/ChatGPT",
                "r/singularity"
            ]
            
            search_terms = [
                "jailbreak",
                "prompt injection",
                "AI safety bypass",
                "adversarial prompt"
            ]
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                for subreddit in subreddits[:3]:  # Limit to avoid rate limits
                    for term in search_terms[:2]:
                        try:
                            # Reddit search endpoint
                            url = f"https://www.reddit.com/{subreddit}/search.json"
                            params = {
                                "q": term,
                                "sort": "new",
                                "limit": 5,
                                "t": "week" if days_back <= 7 else "month"
                            }
                            
                            response = await client.get(
                                url,
                                params=params,
                                headers={"User-Agent": "JailbreakGenomeScanner/1.0"}
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                for post in data.get("data", {}).get("children", [])[:3]:
                                    post_data = post.get("data", {})
                                    
                                    # Check if post is recent enough
                                    created_utc = post_data.get("created_utc", 0)
                                    if created_utc == 0:
                                        continue
                                    post_date = datetime.fromtimestamp(created_utc)
                                    if (datetime.now() - post_date).days > days_back:
                                        continue
                                    
                                    title = post_data.get("title", "")
                                    selftext = post_data.get("selftext", "")[:500]
                                    content = f"{title}\n\n{selftext}"
                                    
                                    events.append(ScrapedEvent(
                                        title=title,
                                        content=content,
                                        source=f"Reddit {subreddit}",
                                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                                        timestamp=post_date,
                                        category="jailbreak",
                                        relevance_score=0.75
                                    ))
                            
                            await asyncio.sleep(2)  # Reddit rate limiting
                        except Exception as e:
                            log.debug(f"Error searching Reddit {subreddit} for {term}: {e}")
        except Exception as e:
            log.debug(f"Reddit search error: {e}")
        
        return events
    
    async def _search_stackexchange(self, days_back: int) -> List[ScrapedEvent]:
        """Search StackExchange sites for jailbreak discussions."""
        events = []
        
        try:
            # StackExchange API (no auth needed, 300 requests/day)
            sites = ["ai.stackexchange.com"]
            
            search_terms = [
                "jailbreak",
                "prompt injection",
                "adversarial",
                "LLM safety"
            ]
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                for site in sites:
                    for term in search_terms[:2]:
                        try:
                            url = "https://api.stackexchange.com/2.3/search/advanced"
                            params = {
                                "site": site.replace(".stackexchange.com", ""),
                                "q": term,
                                "sort": "creation",
                                "order": "desc",
                                "pagesize": 5,
                                "fromdate": int((datetime.now() - timedelta(days=days_back)).timestamp())
                            }
                            
                            response = await client.get(url, params=params)
                            
                            if response.status_code == 200:
                                data = response.json()
                                for item in data.get("items", [])[:3]:
                                    events.append(ScrapedEvent(
                                        title=item.get("title", ""),
                                        content=item.get("body", "")[:500],  # HTML content
                                        source=site,
                                        url=item.get("link", ""),
                                        timestamp=datetime.fromtimestamp(item.get("creation_date", 0)),
                                        category="jailbreak",
                                        relevance_score=0.7
                                    ))
                            
                            await asyncio.sleep(1)
                        except Exception as e:
                            log.debug(f"Error searching StackExchange {site} for {term}: {e}")
        except Exception as e:
            log.debug(f"StackExchange search error: {e}")
        
        return events
    
    def extract_prompts_from_content(self, content: str) -> List[str]:
        """
        Extract jailbreak prompts from scraped content.
        
        Looks for:
        - Code blocks with prompts
        - Quoted text
        - Example sections
        - JSON/YAML structured data
        """
        prompts = []
        
        # Extract from code blocks (``` blocks)
        code_blocks = re.findall(r'```(?:python|json|yaml|text|plain)?\n?(.*?)```', content, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            block = block.strip()
            # Check if it looks like a prompt (contains common prompt patterns)
            if any(keyword in block.lower() for keyword in ["jailbreak", "ignore", "system", "user", "assistant", "role"]):
                if len(block) > 20 and len(block) < 2000:
                    prompts.append(block)
        
        # Extract quoted strings (might be prompts)
        quoted = re.findall(r'"(.*?)"', content)
        for quote in quoted:
            if len(quote) > 30 and len(quote) < 1000:
                # Check if it looks like a prompt
                if any(keyword in quote.lower() for keyword in ["you are", "act as", "ignore", "pretend"]):
                    prompts.append(quote)
        
        # Extract from "Example:" or "Prompt:" sections
        example_sections = re.findall(
            r'(?:example|prompt|jailbreak)[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)',
            content,
            re.IGNORECASE | re.DOTALL
        )
        for section in example_sections:
            section = section.strip()
            if len(section) > 20 and len(section) < 1000:
                prompts.append(section)
        
        # Extract JSON structures that might contain prompts
        json_matches = re.findall(r'\{[^{}]*"prompt"[^{}]*\}', content, re.DOTALL)
        for match in json_matches:
            try:
                data = json.loads(match)
                if "prompt" in data:
                    prompts.append(str(data["prompt"]))
            except:
                pass
        
        # Remove duplicates and clean
        unique_prompts = []
        seen = set()
        for prompt in prompts:
            prompt_clean = prompt.strip()
            if prompt_clean and prompt_clean not in seen and len(prompt_clean) > 20:
                unique_prompts.append(prompt_clean)
                seen.add(prompt_clean)
        
        return unique_prompts[:10]  # Limit to 10 prompts
    
    async def analyze_with_lambda_model(
        self,
        events: List[ScrapedEvent],
        model_endpoint: str
    ) -> List[Dict[str, Any]]:
        """
        Use Lambda-hosted model to analyze scraped events and extract techniques.
        
        Args:
            events: List of scraped events
            model_endpoint: API endpoint of Lambda-hosted model
            
        Returns:
            List of analyzed results with extracted techniques
        """
        analyzed = []
        
        for event in events[:10]:  # Limit analysis
            prompt = f"""Analyze the following content and extract any AI jailbreak techniques, prompt injection methods, or adversarial patterns mentioned.

Title: {event.title}
Content: {event.content}
Source: {event.source}

Extract:
1. Specific jailbreak techniques mentioned
2. Example prompts if provided
3. Target models mentioned
4. Success rate or effectiveness if mentioned

Format as JSON with keys: techniques, example_prompts, target_models, effectiveness."""
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{model_endpoint}/v1/chat/completions",
                        json={
                            "model": "meta-llama/Llama-2-7b-chat-hf",
                            "messages": [
                                {"role": "system", "content": "You are an expert in AI safety and jailbreak techniques."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 500
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        analysis = data["choices"][0]["message"]["content"]
                        
                        analyzed.append({
                            "event": event,
                            "analysis": analysis,
                            "extracted_techniques": self._parse_analysis(analysis)
                        })
            except Exception as e:
                log.error(f"Error analyzing event with Lambda model: {e}")
        
        return analyzed
    
    def _parse_analysis(self, analysis: str) -> List[str]:
        """Parse analysis text to extract techniques."""
        # Simple parsing - in production, use proper JSON parsing
        techniques = []
        lines = analysis.split("\n")
        for line in lines:
            if "technique" in line.lower() or "method" in line.lower():
                techniques.append(line.strip())
        return techniques
    
    async def run_periodic_scraping(
        self,
        interval_hours: int = 6,
        days_back: int = 1
    ) -> None:
        """
        Run periodic scraping in the background.
        
        Args:
            interval_hours: Hours between scraping runs
            days_back: Days back to search each time
        """
        log.info(f"Starting periodic scraping (every {interval_hours} hours)")
        
        while True:
            try:
                events = await self.scrape_recent_events(days_back=days_back)
                log.info(f"Periodic scrape completed: {len(events)} events found")
                
                # Store results (could save to database/file)
                await self._save_scraped_events(events)
                
            except Exception as e:
                log.error(f"Error in periodic scraping: {e}")
            
            # Wait for next interval
            await asyncio.sleep(interval_hours * 3600)
    
    async def _save_scraped_events(self, events: List[ScrapedEvent]) -> None:
        """Save scraped events to file/database."""
        import json
        from pathlib import Path
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        events_data = [{
            "title": e.title,
            "content": e.content[:500],  # Truncate
            "source": e.source,
            "url": e.url,
            "timestamp": e.timestamp.isoformat(),
            "category": e.category,
            "relevance_score": e.relevance_score
        } for e in events]
        
        filepath = data_dir / f"scraped_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        log.info(f"Saved {len(events)} events to {filepath}")
    
    def get_recent_events(self, limit: int = 20) -> List[ScrapedEvent]:
        """Get most recent scraped events."""
        sorted_events = sorted(
            self.scraped_events,
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_events[:limit]
    
    def get_events_by_category(self, category: str) -> List[ScrapedEvent]:
        """Get events filtered by category."""
        return [e for e in self.scraped_events if e.category == category]

