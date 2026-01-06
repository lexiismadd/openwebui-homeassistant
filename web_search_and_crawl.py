"""
title: Web Search and Crawl
description: Search and Crawls the web using SearXNG, OpenWebUI Native Search, and Crawl4AI. Extracts content from URLs using a self-hosted Crawl4AI instance, optionally researching using Crawl4AI Deep Research.
author: lexiismadd
author_url: https://github.com/lexiismadd
funding_url: https://github.com/open-webui
version: 2.7.2
license: MIT
requirements: aiohttp, loguru, crawl4ai, orjson, tiktoken
"""
import traceback
import requests
import orjson
import tiktoken
import aiohttp
import asyncio
from urllib.parse import parse_qs, urlparse, quote
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union, Callable, Literal
from loguru import logger
from crawl4ai import BestFirstCrawlingStrategy, CrawlerRunConfig, DefaultTableExtraction, KeywordRelevanceScorer, LLMConfig, BrowserConfig, CacheMode, DefaultMarkdownGenerator, LLMExtractionStrategy
# from crawl4ai.docker_client import Crawl4aiDockerClient
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# OpenWebUI imports for native search
try:
    from open_webui.main import Request, app # type: ignore
    from open_webui.models.users import UserModel, Users # type: ignore
    from open_webui.routers.retrieval import SearchForm, process_web_search # type: ignore
    NATIVE_SEARCH_AVAILABLE = True
except ImportError:
    NATIVE_SEARCH_AVAILABLE = False
    logger.warning("OpenWebUI native search not available - install requirements or check OpenWebUI version")


class ArticleData(BaseModel):
    topic: str
    summary: str


class Tools:
    class Valves(BaseModel):
        INITIAL_RESPONSE: str = Field(
            title="Initial delta response",
            default="I just need to do a search online to get some more info, I'll get back to you in a minute or so with a response if thats ok with you...",
            description="The response the tool will post in the chat window when it starts its search and crawl. Set as blank for no response."
        )
        USE_NATIVE_SEARCH: bool = Field(
            title="Use Native Search",
            default=True,
            description="Use OpenWebUI's native web search (in addition to or instead of SearXNG).",
        )
        SEARCH_WITH_SEARXNG: bool = Field(
            title="Search with SearXNG",
            default=False,
            description="Use SearXNG for gathering additional URLs for crawling.",
        )
        SEARXNG_BASE_URL: str = Field(
            title="SearXNG Search URL",
            default="http://searxng:8888/search?format=json&q=<query>",
            description="The full URL for your SearXNG API instance. Insert <query> where the search terms should go.",
        )
        SEARXNG_API_TOKEN: str = Field(
            title="SearXNG API Token",
            default="",
            description="The API token or Secret for your SearXNG instance.",
        )
        SEARXNG_METHOD: Literal["GET", "POST"] = Field(
            title="SearXNG HTTP Method",
            default="GET",
            description="HTTP method to use for SearXNG API calls (GET or POST).",
        )
        SEARXNG_TIMEOUT: int = Field(
            title="SearXNG Timeout",
            default=30,
            description="The timeout (in seconds) for SearXNG API requests.",
        )
        SEARXNG_MAX_RESULTS: int = Field(
            title="SearXNG Max Results",
            default=10,
            description="The maximum number of results to return from SearXNG.",
        )
        CRAWL4AI_BASE_URL: str = Field(
            title="Crawl4AI Base URL",
            default="http://crawl4ai:11235",
            description="The base URL for your Crawl4AI instance.",
        )
        CRAWL4AI_USER_AGENT: str = Field(
            title="Crawl4AI User Agent",
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.1.2.3 Safari/537.36",
            description="Custom User-Agent string for Crawl4AI.",
        )
        CRAWL4AI_TIMEOUT: int = Field(
            title="Crawl4AI Timeout",
            default=60,
            description="The timeout (in seconds) for Crawl4AI requests.",
        )
        CRAWL4AI_BATCH: int = Field(
            title="Crawl4AI Batch",
            default=5,
            description="The number of URLs to send to Crawl4AI per batch. If more than this number of URLs are found in total, the tool will send them to Crawl4AI in batches of this number. Useful for reducing the tokens used by the LLM per crawl.",
        )
        CRAWL4AI_MAX_URLS: int = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=20,
            description="The maximum number of URLs to crawl with Crawl4AI.",
        )
        CRAWL4AI_EXTERNAL_DOMAINS: bool = Field(
            title="Crawl External Domains",
            default=False,
            description="Allow Crawl4AI to crawl external/additional URL domains.",
        )
        CRAWL4AI_EXCLUDE_DOMAINS: str = Field(
            title="Excluded Domains",
            default="",
            description="Comma-separated list of external domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS: str = Field(
            title="Excluded Social Media Domains",
            default="facebook.com,twitter.com,x.com,linkedin.com,instagram.com,pinterest.com,tiktok.com,snapchat.com,reddit.com",
            description="Comma-separated list of social media domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_IMAGES: Literal["None", "External", "All"] = Field(
            title="Exclude Images",
            default="None",
            description="Exclude images from crawling (None, External, All).",
        )
        CRAWL4AI_WORD_COUNT_THRESHOLD: int = Field(
            title="Word Count Threshold",
            default=200,
            description="The minimum word count threshold for content to be included.",
        )
        CRAWL4AI_TEXT_ONLY: bool = Field(
            title="Text Only",
            default=False,
            description="Only extract text content, excluding images and other media. (Disables crawling and displaying media in the chat)",
        )
        CRAWL4AI_DISPLAY_MEDIA: bool = Field(
            title="Display Media in Chat",
            default=True,
            description="Display images and videos as clickable links in the chat window.",
        )
        CRAWL4AI_MAX_MEDIA_ITEMS: int = Field(
            title="Max Media Items to Display",
            default=5,
            description="Maximum number of images/videos to display (0 = unlimited).",
        )
        CRAWL4AI_DISPLAY_THUMBNAILS: bool = Field(
            title="Display images as thumbnails",
            default=False,
            description="Display images as thumbnails in the chat window. Turn off to display images full-sized.",
        )
        CRAWL4AI_THUMBNAIL_SIZE: int = Field(
            title="Image thumbnail size",
            default=200,
            description="Image thumbnail size (in px) square.  eg, setting 200 will mean thumbnails are 200x200px in size. Ignored if 'Display images as thumbnails' is off.",
        )
        CRAWL4AI_MAX_TOKENS: int = Field(
            title="Max Tokens used by web content",
            default=0,
            description="Maximum tokens to use for the web search content response. Set to 0 for unlimited.",
        )
        CRAWL4AI_RESEARCH: bool = Field(
            title="Research Mode",
            default=False,
            description="Enable research mode using Crawl4AI with Deep Crawling.",
        )
        CRAWL4AI_RESEARCH_KEYWORD_WEIGHT: float = Field(
            title="Research Keyword Relevance Weight",
            default=0.7,
            description="The keyword relevance weight when using Research mode.",
        )
        CRAWL4AI_RESEARCH_MAX_DEPTH: int = Field(
            title="Research Max Depth",
            default=2,
            le=10,
            description="The maximum depth of links to follow for the Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        CRAWL4AI_RESEARCH_MAX_PAGES: int = Field(
            title="Research Max Pages",
            default=15,
            le=25,
            description="The maximum number of pages to crawl in Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        LLM_BASE_URL: str = Field(
            title="LLM Base URL",
            default="https://openrouter.ai/api/v1",
            description="The base URL for your preferred OpenAI-compatible LLM.",
        )
        LLM_API_TOKEN: str = Field(
            title="LLM API Token",
            default="",
            description="Optional API Token for your preferred OpenAI-compatible LLM.",
        )
        LLM_PROVIDER: str = Field(
            title="LLM Provider and model",
            default="openrouter/@preset/default",
            description="The LLM provider and model to use (see https://docs.crawl4ai.com/core/browser-crawler-config/#3-llmconfig-essentials).",
            examples=[
                "openai/gpt-4o",
                "ollama/llama-3-70b",
                "openrouter/@preset/default",
                "azure/gpt-4o",
                "anthropic/claude-2",
            ],
        )
        LLM_TEMPERATURE: float = Field(
            title="LLM Temperature",
            default=0.3,
            description="The temperature to use for the LLM.",
        )
        LLM_INSTRUCTION: str = Field(
            title="LLM Extraction Instruction",
            default="""Focus on extracting the core content. Summarize lengthy sections into concise points
            Include:
            - Key concepts and explanations
            - Important examples
            - Critical details that enhance understanding
            - Data from tables that support the main content
            - Any relevant data snippets
            Exclude:
            - Navigation elements
            - Sidebars
            - Footer content
            - Marketing or promotional material
            - Advertisements
            - User comments
            - Any other non-essential information
            Format the output as clean markdown with proper code blocks and headers.
            """,
            description="The instruction to use for the LLM when extracting from the webpage.",
        )
        LLM_MAX_TOKENS: int = Field(
            title="LLM Max Tokens",
            default=4096,
            description="The maximum number of tokens to use for the LLM.",
        )
        LLM_TOP_P: float = Field(
            title="LLM Top P",
            default=None,
            description="The top_p value to use for the LLM.",
        )
        LLM_FREQUENCY_PENALTY: float = Field(
            title="LLM Frequency Penalty",
            default=None,
            description="The frequency penalty to use for the LLM.",
        )
        LLM_PRESENCE_PENALTY: float = Field(
            title="LLM Presence Penalty",
            default=None,
            description="The presence penalty to use for the LLM.",
        )

    def __init__(self):
        self.valves = self.Valves()
        if self.valves.SEARCH_WITH_SEARXNG and self.valves.SEARXNG_BASE_URL:
            # Ensure SearXNG URL is properly formatted
            searxng_parsed_url = urlparse(self.valves.SEARXNG_BASE_URL)
            searxng_parsed_url_query = parse_qs(searxng_parsed_url.query)
            if "q" not in searxng_parsed_url_query:
                searxng_parsed_url_query["q"] = ["<query>"]
            if "format" in searxng_parsed_url_query:
                if searxng_parsed_url_query["format"][0] != "json":
                    searxng_parsed_url_query["format"][0] = "json"
            reconstructed_query = "&".join([f"{key}={value[0]}" for key, value in searxng_parsed_url_query.items()])
            self.valves.SEARXNG_BASE_URL = f"{searxng_parsed_url.scheme}://{searxng_parsed_url.netloc}{searxng_parsed_url.path}?{reconstructed_query}"
        
        # Define tools for better LLM integration
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_and_crawl",
                    "description": "Search the web and crawl the resulting pages to extract detailed content with images and videos. Use this for current events, news, research, or any information that needs web search and detailed content extraction. The user can optionally provide specific URLs to include in the crawl.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'latest AI developments', 'Python tutorial')",
                            },
                            "urls": {
                                "type": "array",
                                "description": "Optional list of specific URLs to crawl in addition to search results",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results to crawl (default uses valve setting)",
                                "default": None,
                            },
                            "research_mode": {
                                "type": "boolean",
                                "description": "Enables a special mode called Research Mode which performs deeper web crawling by following links on pages.",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
            
        self.crawl_counter = 0
        self.content_counter = 0
        logger.info("Web Search and Crawl tool initialized")

    async def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


    async def _truncate_content(self, content: str, max_tokens: int, model: str = "gpt-4") -> str:
        """Truncate content to fit within max_tokens."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(content)
        if len(tokens) <= max_tokens:
            return content
        
        # Truncate and add indicator
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text + "\n\n[Content truncated due to length...]"

    async def _validate_image_url(self, url: str) -> bool:
        """
        Validate if an image URL is accessible and returns an image.
        Returns True if valid, False otherwise.
        """
        # if not self.valves.CRAWL4AI_VALIDATE_IMAGES:
        #     return True  # Skip validation if disabled
        
        try:
            timeout = aiohttp.ClientTimeout(total=4)
            url = url.strip()
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                skip_auto_headers={'Accept-Encoding', 'Content-Type'}
            ) as session:
                async with session.head(url, allow_redirects=True) as response:
                    # Check if status is OK
                    if response.status != 200:
                        logger.warning(f"Image validation failed for {url}: Status {response.status}")
                        return False
                    
                    # Check if content-type is an image
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not content_type.startswith('image/'):
                        logger.warning(f"Image validation failed for {url}: Content-Type {content_type}")
                        return False
                    
                    return True
        except asyncio.TimeoutError:
            logger.warning(f"Image validation timeout for {url}")
            return False
        except Exception as e:
            logger.warning(f"Image validation error for {url}: {str(e)}")
            return False
    
    async def _validate_images_batch(self, urls: List[str]) -> List[str]:
        """
        Validate multiple image URLs concurrently.
        Returns list of valid URLs only.
        """
        # if not self.valves.CRAWL4AI_VALIDATE_IMAGES:
        #     return urls  # Skip validation if disabled
        
        tasks = [self._validate_image_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        valid_urls = [url for url, is_valid in zip(urls, results) if is_valid]
        
        if len(valid_urls) < len(urls):
            logger.info(f"Image validation: {len(valid_urls)}/{len(urls)} images are valid")
        
        return valid_urls


    async def get_request(self) -> "Request":
        """Helper to create a request object for native search."""
        if not NATIVE_SEARCH_AVAILABLE:
            raise ImportError("OpenWebUI native search not available")
        return Request(scope={"type": "http", "app": app})
    
    async def _search_native(
        self, 
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None
    ) -> List[str]:
        """Search using OpenWebUI's native web search and return URLs."""
        
        if not self.valves.USE_NATIVE_SEARCH:
            logger.info("Native search is disabled.")
            return []
        
        if not NATIVE_SEARCH_AVAILABLE:
            logger.warning("Native search not available - missing OpenWebUI imports")
            return []
        
        if __user__ is None:
            logger.error("User information required for native search")
            return []
        
        try:
            user = Users.get_user_by_id(__user__["id"])
            if user is None:
                logger.error("User not found")
                return []
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Searching for '{query}'...", "done": False},
                    }
                )
            
            # Use native search
            form = SearchForm.model_validate({"queries": [query]})
            result = await process_web_search(
                request=Request(scope={"type": "http", "app": app}), form_data=form, user=user
            )
            logger.info(f"Native search for '{query}' returned {result}")
            
            urls = [item.get("link") for item in result.get("items", []) if item.get("link")]
            
            logger.info(f"Native search for '{query}' returned {len(urls)} URLs")
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Found {len(urls)} websites...", "done": False},
                    }
                )
            
            return urls
            
        except Exception as e:
            logger.error(f"Error in native search: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Native search encountered an error: {str(e)}", "done": False},
                    }
                )
            return []
    
    async def _search_searxng(self, 
        query: str,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> List[str]:
        """Search SearXNG and return a list of URLs."""
        
        if not self.valves.SEARCH_WITH_SEARXNG:
            logger.info("SearXNG search is disabled.")
            return []
            
        if not self.valves.SEARXNG_BASE_URL:
            logger.error("SearXNG base URL is not configured.")
            return []

        # Use the pre-formatted URL
        url = self.valves.SEARXNG_BASE_URL.replace("<query>", query)
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Add token if configured
        if self.valves.SEARXNG_API_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.SEARXNG_API_TOKEN}"
        
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Searching for '{query}'...", "done": False},
                }
            )
        
        try:
            if self.valves.SEARXNG_METHOD == "POST":
                response = requests.post(
                    url,
                    data={"q": query, "format": "json"},
                    headers=headers,
                    timeout=self.valves.SEARXNG_TIMEOUT
                )
            else:  # GET
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.valves.SEARXNG_TIMEOUT
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract URLs from results
            results = data.get("results", [])
            urls = []
            for result in results[:self.valves.SEARXNG_MAX_RESULTS]:
                if result.get("url"):
                    urls.append(result["url"])
            
            logger.info(f"SearXNG search for '{query}' returned {len(urls)} URLs")
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Found {len(urls)} results...", "done": False},
                    }
                )
            
            return urls
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching SearXNG: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"SearXNG search error: {str(e)}", "done": False},
                    }
                )
            return []
        except Exception as e:
            logger.error(f"Unexpected error in SearXNG search: {str(e)}")
            return []

    async def search_and_crawl(
        self, 
        query: str,
        urls: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        max_images: Optional[int] = None,
        research_mode: Optional[bool] = False,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None
    ) -> Union[list, str]:
        """
        USE THIS TOOL whenever the user asks to 'search' for, 'lookup', 'find' information,
        'browse' the web, 'gather' data on a specific topic, or when any information or data 
        is needed from the internet to respond to the user.
        This tool performs web searches using both Native Search and/or SearXNG to gather relevant URLs,
        then crawls those URLs using Crawl4AI to extract clean content with media.
        :param query: The search query to use.
        :param urls: Optional list of URLs to crawl in addition to those found from searching.
        :param max_results: The maximum number of search results to crawl (per search engine).
        :param max_images: The maximum number of images results to display in the chat window.
        :param research_mode: Enables research mode for deeper web crawling and following links on pages.
        """
        logger.info(f"Starting search and crawl for '{query}'")

        gathered_urls = []
        self.crawl_counter = 0
        self.content_counter = 0
        
        if not max_images:
            max_images = self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
        
        # Add any user-provided URLs first
        if urls:
            for url in urls:
                # Ensure URL starts with http
                if not url.startswith("http"):
                    url = f"https://{url}"
                if url not in gathered_urls:
                    gathered_urls.append(url)
        
        if __event_emitter__ and str(self.valves.INITIAL_RESPONSE).strip() != "":
            await __event_emitter__(
                {
                    "type": "chat:message:delta",
                    "data": {
                        "content": str(self.valves.INITIAL_RESPONSE).strip()
                    },
                }
            )
        # Search with Native Search if enabled
        if self.valves.USE_NATIVE_SEARCH:
            native_urls = await self._search_native(query, __event_emitter__, __user__)
            for url in native_urls:
                if url not in gathered_urls:
                    gathered_urls.append(url)
        
        # Search with SearXNG if enabled
        if self.valves.SEARCH_WITH_SEARXNG:
            searxng_urls = await self._search_searxng(query, __event_emitter__)
            # Apply max_results limit for SearXNG
            max_results = max_results or self.valves.SEARXNG_MAX_RESULTS
            for url in searxng_urls[:max_results]:
                if url not in gathered_urls:
                    gathered_urls.append(url)
        
        # Check if we have URLs to crawl
        if not gathered_urls:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"No URLs found for query '{query}'.", "done": True},
                    }
                )
            logger.info(f"No URLs gathered to crawl for query '{query}'.")
            return f"No URLs found to crawl for the query: {query}."
        
        if len(gathered_urls) > self.valves.CRAWL4AI_MAX_URLS:
            max_urls = max_results or self.valves.CRAWL4AI_MAX_URLS
            gathered_urls = gathered_urls[:max_urls]
            
            
        # Emit status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Found {len(gathered_urls)} URLs. Now crawling...",
                        "done": False,
                    },
                }
            )
        
        # Now crawl all gathered URLs
        crawl_results = []
        batch_count = 1
        image_list = []
        video_list = []
        seen_images = set()
        seen_videos = set()
        total_tokens = 0
        thumbnail_size = self.valves.CRAWL4AI_THUMBNAIL_SIZE or 200
        
        for i in range(0, len(gathered_urls), self.valves.CRAWL4AI_BATCH):
            batch = gathered_urls[i:i + self.valves.CRAWL4AI_BATCH]
            batch_count += 1
            try:
                crawled_batch = await self._crawl_url(
                    urls=batch,
                    query=query,
                    research_mode=research_mode,
                    __event_emitter__=__event_emitter__
                )
                
                logger.info(f"Found {len(crawled_batch.get('content',[]))} content, {len(crawled_batch.get('images',[]))} images, {len(crawled_batch.get('videos',[]))} videos.")
                
                # Compile images
                if crawled_batch.get("images",[]):
                    
                    for img_url in crawled_batch.get("images",[]):
                        parsed_image = urlparse(img_url)
                        base_image_url = f"{parsed_image.scheme}://{parsed_image.netloc}{parsed_image.path}"
                        if base_image_url in seen_images:
                            # Don't display duplicates!
                            continue
                        else:
                            seen_images.add(base_image_url)
                            thumbnail_url = f"https://images.weserv.nl/?url={quote(img_url)}&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                            image_valid = await self._validate_image_url(img_url)
                            thumbnail_valid = await self._validate_image_url(thumbnail_url)
                            if image_valid and thumbnail_valid:
                                # Add if valid
                                image_list.append(img_url)
                            
                # Compile videos
                if crawled_batch.get("videos",[]):
                    
                    for vid_url in crawled_batch.get("videos",[]):
                        parsed_video = urlparse(vid_url)
                        base_video_url = f"{parsed_video.scheme}://{parsed_video.netloc}{parsed_video.path}"
                        if base_video_url in seen_videos:
                            # Don't display duplicates!
                            continue
                        else:
                            seen_videos.add(base_video_url)
                            video_list.append(vid_url)
                            
                
                # Process content, making sure not to exceed the total token count
                data_list = crawled_batch.get("content",[])
                content_list = crawled_batch.get("content",[])
                content_str = orjson.dumps(content_list).decode('utf-8')
                page_tokens = await self._count_tokens(content_str)

                # Check if we need to truncate this page's content
                if self.valves.CRAWL4AI_MAX_TOKENS > 0 and page_tokens > self.valves.CRAWL4AI_MAX_TOKENS:
                    content_str = await self._truncate_content(content_str, self.valves.CRAWL4AI_MAX_TOKENS)
                    # Re-parse the truncated content
                    try:
                        content_list = orjson.loads(content_str.replace("\n\n[Content truncated due to length...]", ""))
                    except:
                        # If parsing fails, use original but truncated
                        pass
                    page_tokens = self.valves.CRAWL4AI_MAX_TOKENS
                    logger.info(f"Truncated content from {url} to {self.valves.CRAWL4AI_MAX_TOKENS} tokens")
                
                    # Check if adding this page would exceed total limit
                    if total_tokens + page_tokens > self.valves.CRAWL4AI_MAX_TOKENS:
                        logger.warning(f"Reached token limit ({self.valves.CRAWL4AI_MAX_TOKENS}). Skipping remaining pages of content.")
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": f"Token limit reached. Processed {len(content_list)} of {len(data_list)} pages.", "done": False},
                                }
                            )
                        continue
                
                total_tokens += page_tokens
                logger.info(f"Page {url}: {page_tokens} tokens (Total: {total_tokens}/{self.valves.CRAWL4AI_MAX_TOKENS if self.valves.CRAWL4AI_MAX_TOKENS > 0 else 'unlimited'})")
                crawl_results.extend(content_list)
                
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_message)
                
        # Display media if enabled
        if __event_emitter__ and self.valves.CRAWL4AI_DISPLAY_MEDIA:
            max_items = self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
            image_list = image_list[:max_images] if max_images > 0 else image_list
            video_list = video_list[:max_items] if max_items > 0 else video_list
            
            # Display images with thumbnails
            if image_list:
                image_markdown = ""
                for img_url in image_list:
                    # Use images.weserv.nl to create thumbnails
                    # Format: https://images.weserv.nl/?url=IMAGE_URL&w=SIZE&h=SIZE&fit=cover
                    if self.valves.CRAWL4AI_DISPLAY_THUMBNAILS:
                        thumbnail_url = f"https://images.weserv.nl/?url={quote(img_url)}&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                    else:
                        thumbnail_url = img_url
                    # Wrap thumbnail in link to original
                    image_markdown += f"[![image]({thumbnail_url})]({img_url})\n"
                    
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": image_markdown},
                    }
                )
            # Display videos as links
            if video_list:
                video_markdown = f"\n\n*Videos links:*\n"
                # Format as markdown for clickable links (videos can't embed easily)
                for idx, vid_url in enumerate(video_list, 1):
                    video_markdown += f"{idx}. [{vid_url}]({vid_url})\n"
                
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": video_markdown},
                    }
                )
        
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Completed crawling {len(crawl_results)} web pages.", "done": True},
                }
            )
        
        return crawl_results




    async def _crawl_url(self, 
        urls: Union[list, str],
        query: Optional[str] = None,
        research_mode: bool = False,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> dict:
        """
        Internal function to crawl URLs and extract content.
        This tool converts any webpage into clean content and extracts images and videos.
        :param urls: The exact web URL(s) to extract data from.
        :param query: Optional search query for research mode.
        :param research_mode: Enable research mode with deep crawling.
        """
        if isinstance(urls, str):
            urls = [urls]
            
        for idx, url in enumerate(urls):
            # Ensure URL starts with http
            if not url.startswith("http"):
                urls[idx] = f"https://{url}"

        endpoint = f"{self.valves.CRAWL4AI_BASE_URL}/crawl"
        
        logger.info(f"Using LLM provider: {self.valves.LLM_PROVIDER}")
        
        # Building configs
        browser_config = BrowserConfig(
            headless=True,
            light_mode=True,
            headers={
                "sec-ch-ua": '"Chromium";v="116", "Not_A Brand";v="8", "Google Chrome";v="116"'
            },
            extra_args=[
                "--no-sandbox",
                "--disable-gpu",
            ],
        )
        
        llm_config = LLMConfig(
            provider=self.valves.LLM_PROVIDER,
            base_url=self.valves.LLM_BASE_URL,
            temperature=self.valves.LLM_TEMPERATURE or 0.3,
            max_tokens=self.valves.LLM_MAX_TOKENS or None,
            top_p=self.valves.LLM_TOP_P or None,
            frequency_penalty=self.valves.LLM_FREQUENCY_PENALTY or None,
            presence_penalty=self.valves.LLM_PRESENCE_PENALTY or None
        )
        if self.valves.LLM_API_TOKEN:
            llm_config.api_token = self.valves.LLM_API_TOKEN

        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            instruction=self.valves.LLM_INSTRUCTION,
            input_format="fit_markdown",
            schema=ArticleData.model_json_schema(),
        )
        
        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(),
            options={
                "ignore_links": True,
                "escape_html": False,
                "body_width": 80
            }
        )
        
        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            extraction_strategy=extraction_strategy,
            table_extraction=DefaultTableExtraction(),
            exclude_external_links=not self.valves.CRAWL4AI_EXTERNAL_DOMAINS,
            exclude_social_media_domains=[d.strip() for d in self.valves.CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS.split(",") if d.strip()],
            exclude_domains=[d.strip() for d in self.valves.CRAWL4AI_EXCLUDE_DOMAINS.split(",") if d.strip()],
            user_agent=self.valves.CRAWL4AI_USER_AGENT,
            stream=False,
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.valves.CRAWL4AI_TIMEOUT * 1000,  # Convert to milliseconds
            only_text=self.valves.CRAWL4AI_TEXT_ONLY,
            word_count_threshold=self.valves.CRAWL4AI_WORD_COUNT_THRESHOLD,
            exclude_all_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "All",
            exclude_external_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "External",
        )
        
        # Add research mode configuration if enabled
        if (self.valves.CRAWL4AI_RESEARCH or research_mode) and query:
            
            # Create a relevance scorer
            keywords = query.split()
            keyword_scorer = KeywordRelevanceScorer(
                keywords=keywords,
                weight=self.valves.CRAWL4AI_RESEARCH_KEYWORD_WEIGHT or 0.7,
                case_sensitive=False
            )

            crawler_config.deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=self.valves.CRAWL4AI_RESEARCH_MAX_DEPTH or 2,
                url_scorer=keyword_scorer,
                include_external=self.valves.CRAWL4AI_EXTERNAL_DOMAINS,
                max_pages=self.valves.CRAWL4AI_RESEARCH_MAX_PAGES or 15,
            )

        if (self.valves.CRAWL4AI_RESEARCH or research_mode) and query:
            if len(urls) > 1:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Researching across {len(urls)} websites...", "done": False},
                        }
                    )
            else:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Researching {urls[0]}...", "done": False},
                        }
                    )

        self.crawl_counter += len(urls)
        
        logger.info(f"Contacting Crawl4AI at {endpoint} for URLs: {urls}")

        headers = {"Content-Type": "application/json"}

        payload = {
            "urls": urls,
            "browser_config": browser_config.dump(),
            "crawler_config": crawler_config.dump()
        }

        try:
            # Using a timeout to prevent the UI from hanging
            timeout = self.valves.CRAWL4AI_TIMEOUT*len(urls) + 60
            # logger.info(f"Payload {payload}")
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            # if __event_emitter__:
            #     await __event_emitter__(
            #         {
            #             "type": "status",
            #             "data": {"description": f"Analysing the content...", "done": False},
            #         }
            #     )
                
            # Crawl4AI returns content in the 'results' key as a list
            results = []
            seen_images = set()
            seen_videos = set()
            all_images = []
            all_videos = []
            
            # logger.info(f"Received {data}")
            data_list = data.get("results", [])
            for item in data_list:
                if item.get("success") is not True:
                    continue
                    
                url = item.get("url", "")
                parsed_url = urlparse(url)
                
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Analyzing {url}...", "done": False},
                        }
                    )
                
                # Extract media
                image_list = []
                # logger.info(f"Images found: {len(item.get('media', {}).get('images', []))}")
                found_images = list(filter(lambda x: x.get("score", 0) >= 5, item.get("media", {}).get("images", [])))
                for img in found_images:
                    src = img.get("src")
                    # logger.info(f"Processing image: {src}")
                    if src:
                        # Fix protocol-relative URLs
                        if src.startswith("//"):
                            src = f"https:{src}"
                        elif not src.startswith("http"):
                            # Handle relative URLs
                            src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"
                        # logger.info(f"as image: {src}")
                        parsed_image = urlparse(src)
                        if f"{parsed_image.scheme}://{parsed_image.netloc}/{parsed_image.path}" not in seen_images:
                            seen_images.add(f"{parsed_image.scheme}://{parsed_image.netloc}/{parsed_image.path}")
                            image_list.append(src)
                            # logger.info(f"Image: {src}")
                
                video_list = []
                # logger.info(f"Videos found: {len(item.get('media', {}).get('videos', []))}")
                found_videos = list(filter(lambda x: x.get("score", 0) >= 5, item.get("media", {}).get("videos", [])))
                for vid in found_videos:
                    src = vid.get("src")
                    if src:
                        # Fix protocol-relative URLs
                        if src.startswith("//"):
                            src = f"https:{src}"
                        elif not src.startswith("http"):
                            # Handle relative URLs
                            src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"
                        parsed_video = urlparse(src)
                        if f"{parsed_video.scheme}://{parsed_video.netloc}/{parsed_video.path}" not in seen_images:
                            seen_videos.add(f"{parsed_video.scheme}://{parsed_video.netloc}/{parsed_video.path}")
                            video_list.append(src)

                
                await __event_emitter__(
                    {
                        "type": "files",  # or "chat:message:files"
                        "data": {
                            "files": image_list + video_list
                        },
                    }
                )
                # Extract content
                tmp_content = orjson.loads(item.get("extracted_content", []))
                content_list = [
                    {"topic": item["topic"], "summary": item["summary"]} 
                    for item in tmp_content 
                    if item.get("error") is False
                ]
                
                # Build result with URL included
                results.append({
                    "url": url,  # Source URL for citation
                    "title": item.get("metadata", {}).get("title", ""),
                    "content": content_list,
                    "images": image_list,
                    "videos": video_list
                })
                all_images.extend(image_list)
                all_videos.extend(video_list)
                
                # Emit citation for this URL
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [f"Content from {url}"],
                                "metadata": [{"source": url}],
                                "source": {"name": item.get("metadata", {}).get("title", url)},
                            },
                        }
                    )
                
            self.content_counter += len(results)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Analyzed {self.content_counter} page{'s' if self.content_counter > 1 else ''} of {self.crawl_counter} URL{'s' if self.crawl_counter > 1 else ''}...", "done": False},
                    }
                )
            
            logger.info(f"Successfully crawled {len(results)} URLs")
            response = {
                "content": results,
                "images": all_images or [],
                "videos": all_videos or []
            }
            return response

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error connecting to Crawl4AI: {str(e)}. Check if the URL {self.valves.CRAWL4AI_BASE_URL} is accessible."
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return {"error": error_msg, "details": e}
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return {"error": error_msg, "details": e}