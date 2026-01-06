# 🌐 Web Search and Crawl for Open WebUI

**Version:** 2.7.2

**Author:** lexiismadd

This tool enables your Open WebUI instance to not only search the internet but to **deeply crawl, extract, and summarize** content from web pages. It combines the power of search engines (SearXNG and Open WebUI Native Search) with the advanced extraction capabilities of **Crawl4AI**.

---

## ✨ Key Features

* **Dual-Engine Search:** Simultaneously utilizes **SearXNG** and **OpenWebUI Native Search** to find the most relevant URLs.
* **Intelligent Crawling:** Powered by [Crawl4AI](https://crawl4ai.com), it extracts clean, markdown-formatted content while stripping away ads, sidebars, and navigation clutter.
* **Research Mode:** Recursively follows links on discovered pages to perform "Deep Research" based on keyword relevance using **Crawl4AI**'s **Deep Crawl** feature.
* **LLM-Driven Extraction:** Uses an OpenAI-compatible LLM (like GPT-4, Claude, or Ollama) to summarize and structure the crawled data before it reaches your chat.
* **Media Enrichment:** Automatically identifies and displays high-quality images and videos from the sources, providing clickable thumbnails directly in the chat.
* **Smart Token Management:** Includes automatic content truncation and token counting (via `tiktoken`) to ensure responses stay within model limits.
* **Concurrent Validation:** Validates image URLs in batches to ensure only "live" media is displayed.
* **Highly Configurable:** Offers many configuration valves to tune the tool to your preference and environment.

---

## 🚀 How It Works

1. **Trigger:** When you ask a question requiring real-time data, the tool is called.
2. **Gathering:** It queries SearXNG or Native Search and combines those results with any specific URLs you provided.
3. **Crawling:** It sends those URLs to a self-hosted **Crawl4AI** instance.
4. **Extraction:** The LLM configured in the "Valves" processes the raw HTML/Markdown into a structured summary focusing on your core instructions.
5. **Delivery:** The tool returns a clean list of summaries, citations, and a gallery of relevant media to the chat interface.

---

## ⚙️ Configuration Valves

| Category | Key | Description |
| --- | --- | --- |
| **General** | `Initial delta response` | The message shown in chat when the tool starts working. |
| **Search** | `Use Native Search` | Enable/Disable OpenWebUI's internal search. |
| **Search** | `Search with SearXNG` | Enable/Disable SearXNG integration. |
| **Search** | `SearXNG Search URL` | SearXNG Search query URL. |
| **Search** | `SearXNG API Token` | The API token or Secret for your SearXNG instance. |
| **Search** | `SearXNG HTTP Method` | HTTP method to use for SearXNG API calls (GET or POST). |
| **Search** | `SearXNG Timeout` | The timeout (in seconds) for SearXNG API requests. |
| **Crawl4AI** | `Crawl4AI Base URL` | The URL of your Crawl4AI Docker/Server instance. |
| **Crawl4AI** | `Crawl4AI User Agent` | Custom User-Agent string for Crawl4AI. |
| **Crawl4AI** | `Crawl4AI Timeout` | The timeout (in seconds) for Crawl4AI requests. |
| **Crawl4AI** | `Crawl4AI Batch` | The number of URLs to send to Crawl4AI per batch. |
| **Crawl4AI** | `Crawl4AI Maximum URLs to crawl` | The maximum number of URLs to crawl with Crawl4AI. |
| **Crawl4AI** | `Crawl External Domains` | Allow Crawl4AI to crawl external/additional URL domains. |
| **Crawl4AI** | `Excluded Domains` | Comma-separated list of external domains to exclude from crawling. |
| **Crawl4AI** | `Excluded Social Media Domains` | Comma-separated list of social media domains to exclude from crawling. |
| **Crawl4AI** | `Exclude Images` | Exclude images from crawling. |
| **Crawl4AI** | `Word Count Threshold` | The word count threshold for content to be included. |
| **Crawl4AI** | `Text Only` | Only extract text content, excluding images and other media. |
| **Crawl4AI** | `Display Media in Chat` | Display images and videos as clickable links in the chat window. |
| **Crawl4AI** | `Max Media Items to Display` | Maximum number of images/videos to display. |
| **Crawl4AI** | `Display images as thumbnails` | Display images as thumbnails in the chat window. |
| **Crawl4AI** | `Image thumbnail size` | Image thumbnail size (in px) square. |
| **Crawl4AI** | `Max Tokens used by web content` | Maximum tokens to use for the web search content response. |
| **Research** | `Research Mode` | Toggles "Research Mode" for deep crawling and multi-layered link following. |
| **Research** | `Keyword Relevance Weight` | The keyword relevance weight when using Research mode. |
| **Research** | `Max Depth` | The maximum depth of links to follow for the Research mode. |
| **Research** | `Max Pages` | The maximum number of pages to crawl in Research mode. |
| **LLM** | `LLM Base URL` | The base URL for your preferred OpenAI-compatible LLM. |
| **LLM** | `LLM API Token` | Optional API Token for your preferred OpenAI-compatible LLM. |
| **LLM** | `LLM Provider` | The LLM provider and model to use (e.g., `openai/gpt-4o`). |
| **LLM** | `LLM Temperature` | The temperature to use for the LLM. |
| **LLM** | `LLM Extraction Instruction` | The instruction to use for the LLM when extracting from the webpage. |
| **LLM** | `LLM Max Tokens` | The maximum number of tokens to use for the LLM. |
| **LLM** | `LLM Top P` | The top_p value to use for the LLM. |
| **LLM** | `LLM Frequency Penalty` | The frequency penalty to use for the LLM. |
| **LLM** | `LLM Presence Penalty` | The presence penalty to use for the LLM. |

---

## 🛠️ Requirements

To use this tool, your environment must have:

* **Crawl4AI Server:** A running instance of Crawl4AI (usually via Docker).
* **OpenWebUI:** A recent version (v0.6.42 or higher) to support Native Search and Tools.

---

## 📖 Usage Example

**Standard Search:**

> "Find the latest news on SpaceX Starship and summarize the key findings."

**Targeted Crawl:**

> "Crawl [https://example.com](https://example.com) and tell me their pricing structure."

**Research Mode:**

> "Perform a deep research search on 'Ambient Computing' and find at least 10 sources."

---

## 📝 License

This project is licensed under the **MIT License**.
