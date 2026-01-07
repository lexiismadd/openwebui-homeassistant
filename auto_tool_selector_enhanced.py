"""
title: Auto Tool Selector Enhanced
author: lexiismadd
author_url: https://github.com/lexiismadd
version: 2.0.0
description: >
    Automatically selects relevant tools for a user query based on conversation
    history and available tool metadata. Works with Open WebUI 0.6.43.
    Supports both built-in features and custom tools, avoiding duplicates.
    Install via Functions in Open WebUI.
    Forked and updated from original Auto Tool Selector by siwadon-jay (https://github.com/siwadon-jay)
required_open_webui_version: 0.6.43
requirements: aiohttp, loguru, orjson
"""

import traceback
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Literal, Optional, Dict, List
import json
import orjson
import re
import ast
import logging
from difflib import SequenceMatcher
from string import Template

from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.utils.task import get_task_model_id

logger = logging.getLogger(__name__)

STATUS_MESSAGES = {
    "analyse_query": {
        "simple": "Thinking...",
        "friendly": "Thinking what tools I can use to help with a response for you...",
        "professional": "Analyzing query to select relevant tools..."
    },
    "no_tool_available": {
        "simple": "No tools available",
        "friendly": "There are no tools available for me to use... Should there be?",
        "professional": "Unable to identify any tools..."
    },
    "prepare_tools": {
        "simple": Template("Thinking...."),
        "friendly": Template("I think I found $count tools that might help..."),
        "professional": Template("Analyzing $count available tools...")
    },
    "features_tools_msg": {
        "simple": Template("Thinking....."),
        "friendly": Template("Good news! I found $parts that I can use..."),
        "professional": Template("Selected $parts: $tool_ids")
    },
    "no_tool_found": {
        "simple": "No relevant tools found",
        "friendly": "I'm sorry, but i can't seem to find a suitable tool to help answer your query",
        "professional": "No relevant tools found for this query"
    }
}


class Filter:
    """
    Automatically selects relevant tools for a user query based on:
    - Conversation history
    - Available tools and built-in features
    Filters out duplicate or highly similar tools and updates the workflow context.
    """

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority order for filter execution. Lower values run first."
        )
        template: str = Field(
            default="""Available Tools and Features:
{{TOOLS}}

Analyze the user's query and conversation history to select ALL relevant tools and features that would be helpful.

Guidelines:
- Select tools that directly address the user's request
- Consider the context from conversation history
- You can select multiple tools if they work together
- Built-in features (web_search, image_generation, code_interpreter) should be selected when appropriate
- Custom tools should be selected based on their descriptions
- Avoid selecting tools with nearly identical functionality
- If unsure, lean towards selecting tools that might be helpful

Return ONLY a valid JSON array of tool/feature IDs. No explanation, no markdown, just the array.
Examples: ["web_search_and_crawl", "image_generation", "chat_history_search"] or [] if no tools match.

Important: Return an empty array [] if no tools are relevant, not null or undefined.""",
            description="System prompt template for tool selection. Use {{TOOLS}} placeholder."
        )
        status: bool = Field(
            default=True,
            description="Show status updates during tool selection"
        )
        debug: bool = Field(
            default=False,
            description="Enable detailed debug logging"
        )
        max_tools: int = Field(
            default=50,
            ge=1,
            le=100,
            description="Maximum number of tools to consider"
        )
        similarity_threshold: float = Field(
            default=0.8,
            ge=0.5,
            le=1.0,
            description="Threshold for filtering similar tools (0.5-1.0)"
        )
        max_history_chars: int = Field(
            default=1000,
            ge=100,
            le=5000,
            description="Maximum characters from conversation history to include"
        )
        enable_builtin_websearch: bool = Field(
            default=True,
            description="Enable automatic selection of built-in web search"
        )
        enable_builtin_image_generation: bool = Field(
            default=True,
            description="Enable automatic selection of built-in image generation"
        )
        enable_builtin_code_interpreter: bool = Field(
            default=True,
            description="Enable automatic selection of built-in code interpreter"
        )
        enable_other_builtin_features: bool = Field(
            default=True,
            description="Enable automatic selection of other built-in features not listed above"
        )
        enable_custom_tools: bool = Field(
            default=True,
            description="Enable automatic selection of custom tools"
        )
        exclude_custom_tools: str = Field(
            default="",
            description="Comma-separated list of tool IDs to exclude from automatic selection. Leave blank to automatically select from all available custom tools."
        )
        force_custom_tools: str = Field(
            default="",
            description="Comma-separated list of tool IDs to force the use of."
        )
        preferred_task_model: str = Field(
            default="",
            description="Preferred model ID for tool selection (e.g., 'openai/gpt-4o-mini', 'llama3.2:3b'). Leave blank to use Open WebUI's configured task models or fall back to your current model."
        )
        status_messages: Literal["simple","friendly","professional"] = Field(
            default="simple",
            description="What type of tone should the status messages from the function use?"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.s_key = self.valves.status_messages if self.valves.status_messages else "simple" # Status messages key
        logger.info(f"Auto Tool Selector Instantiated")

    # -------------------- Helper: Event emitter --------------------
    async def emit_status(
        self,
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        step: str,
        description: str,
        done: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit status updates to the UI"""
        if self.valves.status or self.valves.debug:
            data = {"step": step, "description": description, "done": done}
            if extra:
                data.update(extra)
            await __event_emitter__({"type": "status", "data": data})

    # -------------------- Helper: Prepare tools --------------------
    def prepare_tools(
        self, __model__: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Gather all available tools and built-in features.
        Returns a list of dicts with 'id', 'description', and 'type' keys.
        """
        all_tools: List[Dict[str, str]] = []

        # Add built-in features if enabled
        if self.valves.enable_builtin_websearch or self.valves.enable_builtin_image_generation or self.valves.enable_builtin_code_interpreter:
            built_in_features = dict()
            if self.valves.enable_builtin_websearch:
                built_in_features["web_search"] = "Search the internet for current information, news, facts, or research"
            if self.valves.enable_builtin_image_generation:
                built_in_features["image_generation"] = "Generate, create, or produce images based on text descriptions"
            if self.valves.enable_builtin_code_interpreter:
                built_in_features["code_interpreter"] = "Execute Python code, analyze data, create visualizations, or perform calculations"

            all_tools.extend([
                {"id": k, "description": v, "type": "builtin"}
                for k, v in built_in_features.items()
            ])

        # Add custom tools if enabled
        if self.valves.enable_custom_tools:
            try:
                custom_tools = Tools.get_tools()
                custom_tool_exclusions = [str(tool_id).strip().lower() for tool_id in self.valves.exclude_custom_tools.split(",")]
                for tool in custom_tools:
                    tool_id = tool.id
                    if str(tool_id).lower() in custom_tool_exclusions:
                        # Don't include
                        continue
                    # Get description from meta or use default
                    description = ""
                    if hasattr(tool, "meta") and tool.meta:
                        if hasattr(tool.meta, "description"):
                            description = tool.meta.description
                        elif isinstance(tool.meta, dict):
                            description = tool.meta.get("description", "")
                    
                    # Fallback to name if no description
                    if not description and hasattr(tool, "name"):
                        description = f"Custom tool: {tool.name}"
                    elif not description:
                        description = f"Custom tool: {tool_id}"
                    
                    all_tools.append({
                        "id": tool_id,
                        "description": description,
                        "type": "custom"
                    })
            except Exception as e:
                logger.error(f"Error loading custom tools: {e}")
                if self.valves.debug:
                    logger.exception(f"Full traceback:{traceback.format_exc()}")

        # Filter by model's allowed tools if specified
        if __model__ and __model__.get("info", {}).get("meta", {}).get("toolIds"):
            available_tool_ids: List[str] = __model__["info"]["meta"]["toolIds"]
            all_tools = [t for t in all_tools if t["id"] in available_tool_ids]

        # Limit to max_tools
        return all_tools[: self.valves.max_tools]

    # -------------------- Helper: Summarize history --------------------
    @staticmethod
    def summarize_history(messages: List[Dict[str, Any]], max_chars: int = 1000) -> str:
        """
        Create a concise summary of recent conversation history.
        """
        # Get last 10-15 messages
        last_messages: List[Dict[str, Any]] = messages[-15:]
        summary_lines: List[str] = []

        for msg in last_messages:
            role: str = msg.get("role", "").upper()
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                # Extract text from content array
                content_text = " ".join([
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ])
            else:
                content_text = str(content)
            
            content_text = content_text.replace("\n", " ").strip()
            
            if content_text and role in ["USER", "ASSISTANT"]:
                # Truncate individual messages if too long
                if len(content_text) > 200:
                    content_text = content_text[:200] + "..."
                summary_lines.append(f"{role}: {content_text}")

        summary_text: str = " | ".join(summary_lines)
        
        # Truncate from the start (keep most recent context)
        if len(summary_text) > max_chars:
            summary_text = "..." + summary_text[-(max_chars - 3):]
        
        return summary_text

    # -------------------- Helper: Parse model response --------------------
    @staticmethod
    def parse_model_response(content: str) -> List[str]:
        """
        Extract tool IDs from the model's response.
        Handles various formats including JSON arrays and lists.
        """
        selected_tool_ids: List[str] = []
        
        if not content or not isinstance(content, str):
            return selected_tool_ids

        # Clean the content
        content = content.strip()
        
        # Try to find JSON array patterns
        # Match [...] including nested structures
        matches: List[str] = re.findall(r'\[[\s\S]*?\]', content)
        
        for raw_list in matches:
            # Try JSON first
            try:
                parsed = json.loads(raw_list)
                if isinstance(parsed, list):
                    # Filter to only strings
                    return [str(item).strip() for item in parsed if item]
            except json.JSONDecodeError:
                pass
            
            # Try literal_eval as fallback
            try:
                parsed = ast.literal_eval(raw_list)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if item]
            except (ValueError, SyntaxError):
                pass
        
        # If no valid array found, try to extract quoted strings
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', content)
        if quoted_matches:
            return [match.strip() for match in quoted_matches]
        
        return selected_tool_ids

    # -------------------- Helper: Filter similar tools --------------------
    @staticmethod
    def filter_similar_tools(
        tool_ids: List[str],
        available_tools: List[Dict[str, str]],
        threshold: float = 0.8,
    ) -> List[str]:
        """
        Remove tools with very similar names/descriptions to avoid redundancy.
        Uses sequence matching on tool descriptions.
        """
        if not tool_ids or threshold >= 1.0:
            return tool_ids

        filtered: List[str] = []
        
        # Create lookup for tool descriptions
        tool_lookup: Dict[str, str] = {
            t["id"]: (t.get("description", "") or t["id"]).lower()
            for t in available_tools
        }
        
        seen_descriptions: List[str] = []

        for tid in tool_ids:
            description = tool_lookup.get(tid, tid.lower())
            
            # Check similarity against already selected tools
            is_similar = any(
                SequenceMatcher(None, description, seen).ratio() >= threshold
                for seen in seen_descriptions
            )
            
            if not is_similar:
                filtered.append(tid)
                seen_descriptions.append(description)
            elif threshold > 0.9:  # Very high threshold, might want to keep it anyway
                filtered.append(tid)

        return filtered

    # -------------------- Main inlet function --------------------
    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __request__: Any,
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main filter function that processes incoming requests and selects appropriate tools.
        """
        try:
            self.s_key = self.valves.status_messages if self.valves.status_messages else "simple"
            messages: List[Dict[str, Any]] = body.get("messages", [])
            user_message: Optional[str] = get_last_user_message(messages)
            logger.info("Starting Auto Tool Selection")
            logger.info(f"    └ tone: {self.s_key}")
            if self.valves.debug:
                logger.info(f"    └ body: {json.dumps(body)}")
            # Early exit if no user message
            if not user_message:
                if self.valves.debug:
                    logger.info("No user message found, skipping tool selection")
                return body

            await self.emit_status(
                __event_emitter__, 
                "start", 
                STATUS_MESSAGES.get("analyse_query",{}).get(self.s_key)
            )

            # Step 1: Prepare available tools
            available_tools: List[Dict[str, str]] = self.prepare_tools(__model__)
            
            if not available_tools:
                status_message = STATUS_MESSAGES.get("no_tool_available",{}).get(self.s_key)
                if self.valves.debug:
                    logger.warning("No tools available for selection")
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    status_message,
                    done=True
                )
                return body

            tool_descriptions: str = "\n".join([
                f"- {t['id']} ({t.get('type', 'unknown')}): {t['description']}"
                for t in available_tools
            ])
            
            system_prompt: str = self.valves.template.replace(
                "{{TOOLS}}", tool_descriptions
            )
            status_message = STATUS_MESSAGES.get("prepare_tools",{}).get(self.s_key).substitute(count=len(available_tools))
            await self.emit_status(
                __event_emitter__,
                "prepare_tools",
                status_message
            )
            

            # Step 2: Summarize conversation history
            summary_history: str = self.summarize_history(
                messages, self.valves.max_history_chars
            )

            if self.valves.debug:
                logger.info(f"History summary ({len(summary_history)} chars): {summary_history[:200]}...")

            # Step 3: Get task model for tool selection
            logger.info("Step 3: Determining task model")
            user_id: Optional[str] = __user__.get("id") if __user__ else None
            user_obj: Optional[Users] = None
            if user_id:
                try:
                    user_obj = Users.get_user_by_id(user_id)
                    logger.info(f"Retrieved user object for user_id: {user_id}")
                except Exception as e:
                    logger.warning(f"Could not fetch user object: {e}")
                    logger.exception(f"Traceback:\n{traceback.format_exc()}")

            model_id: Optional[str] = body.get("model")
            models: Dict[str, Any] = getattr(__request__.app.state, "MODELS", {})
            
            # Check if the current model is a workspace model (custom preset)
            # Workspace models have an "info" structure that contains the base model ID
            current_model_info = models.get(model_id, {})
            
            # Try to determine the actual base model being used
            # Workspace models wrap a base model, so we need to check the base
            base_model_id = model_id
            is_workspace_model = False
            
            if current_model_info:
                # Check if this is a workspace model with base_model_id
                info = current_model_info.get("info", {})
                if info and "base_model_id" in info:
                    base_model_id = info["base_model_id"]
                    is_workspace_model = True
                    logger.info(f"Detected workspace model '{model_id}' wrapping base model '{base_model_id}'")
                elif info and "id" in info and info["id"] != model_id:
                    # Some workspace models might store base model in 'id'
                    base_model_id = info["id"]
                    is_workspace_model = True
                    logger.info(f"Detected workspace model '{model_id}' with base model '{base_model_id}'")
            
            # Get base model info for later fallback
            base_model_info = models.get(base_model_id, current_model_info)
            
            logger.info(f"Current model: {model_id}, Base model: {base_model_id}, Is workspace: {is_workspace_model}")
            
            # Get configured task models
            task_model_config_local = __request__.app.state.config.TASK_MODEL
            task_model_config_external = __request__.app.state.config.TASK_MODEL_EXTERNAL
            
            logger.info(f"Task model configs - Local: '{task_model_config_local}', External: '{task_model_config_external}'")

            
            # ============================================================
            # PRIORITY ORDER FOR TASK MODEL SELECTION
            # ============================================================
            # 1. Manually specified model (in valves) -> if specified and available, use
            # 2. Local task model -> if available, use
            # 3. External task model -> if available, use
            # 4. Workspace base model -> if current is workspace and base available, use
            # 5. Current base model -> final fallback
            # ============================================================
            
            task_model_id = None
            selection_reason = ""
            
            # PRIORITY 1: Manually specified model in valves (highest priority)
            if self.valves.preferred_task_model and self.valves.preferred_task_model.strip():
                preferred = self.valves.preferred_task_model.strip()
                if preferred in models:
                    task_model_id = preferred
                    selection_reason = "manually specified in valves"
                    logger.info(f"✓ Priority 1: Using preferred task model from valve: {task_model_id}")
                else:
                    logger.warning(f"✗ Priority 1: Preferred task model '{preferred}' not found in available models, moving to next priority")
            
            # PRIORITY 2: Local task model (if configured and available)
            if not task_model_id and task_model_config_local:
                if task_model_config_local in models:
                    task_model_id = task_model_config_local
                    selection_reason = "configured local task model"
                    logger.info(f"✓ Priority 2: Using configured LOCAL task model: {task_model_id}")
                else:
                    logger.warning(f"✗ Priority 2: Local task model '{task_model_config_local}' not found, moving to next priority")
            
            # PRIORITY 3: External task model (if configured and available)
            if not task_model_id and task_model_config_external:
                if task_model_config_external in models:
                    task_model_id = task_model_config_external
                    selection_reason = "configured external task model"
                    logger.info(f"✓ Priority 3: Using configured EXTERNAL task model: {task_model_id}")
                else:
                    logger.warning(f"✗ Priority 3: External task model '{task_model_config_external}' not found, moving to next priority")
            
            # PRIORITY 4: Workspace base model (if current is workspace and base is available)
            if not task_model_id and is_workspace_model:
                if base_model_id and base_model_id in models and base_model_id != model_id:
                    task_model_id = base_model_id
                    selection_reason = "workspace model's base model"
                    logger.info(f"✓ Priority 4: Using workspace base model: {task_model_id}")
                else:
                    logger.warning(f"✗ Priority 4: Workspace base model '{base_model_id}' not available, moving to next priority")
            
            # PRIORITY 5: Current base model (final fallback)
            if not task_model_id:
                if base_model_id and base_model_id in models:
                    task_model_id = base_model_id
                    selection_reason = "current base model (fallback)"
                    logger.info(f"✓ Priority 5: Using current base model as fallback: {task_model_id}")
                else:
                    logger.error(f"✗ Priority 5: Current base model '{base_model_id}' not available")
            
            # Final safety check - if still no valid model, error out
            if not task_model_id or task_model_id not in models:
                logger.error(f"FAILED: No valid task model found after checking all priorities")
                logger.error(f"Current model: {model_id}, Base model: {base_model_id}")
                logger.error(f"Available models: {list(models.keys())}")
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    "No valid model available for tool selection. Please configure a task model in Admin Settings.",
                    done=True,
                )
                return body

            logger.info(f"Task model selected: {task_model_id} ({selection_reason})")

            # Step 4: Create selection prompt
            prompt: str = f"""Conversation Context:
{summary_history}

Current User Query:
{user_message}

Task: Select all relevant tools and features that would help address this query."""

            payload: Dict[str, Any] = {
                "model": task_model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": 0.1,  # Lower temperature for more consistent tool selection
            }

            # Step 5: Call model to select tools
            logger.info("Step 5: Calling model for tool selection")
            try:
                response = await generate_chat_completion(
                    request=__request__, form_data=payload, user=user_obj
                )
                
                # Extract the response body from JSONResponse
                if hasattr(response, 'body'):
                    import json as json_lib
                    response_data = json_lib.loads(response.body)
                    logger.info("Parsed JSONResponse.body")
                else:
                    response_data = response
                    logger.info("Using response directly as dict")
                
            except Exception as e:
                logger.error(f"Error calling generate_chat_completion: {e}")
                logger.exception(f"Traceback:\n{traceback.format_exc()}")
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    f"Error calling model for tool selection: {str(e)}",
                    done=True,
                )
                return body
            
            # Check for error in response
            if "error" in response_data:
                error_msg = response_data["error"].get("message", "Unknown error")
                error_details = response_data["error"]
                logger.error(f"Model API returned error: {error_msg}")
                logger.error(f"Error details: {error_details}")
                
                await self.emit_status(
                    __event_emitter__,
                    "error",
                    f"Model error: {error_msg}. Check that your task model is properly configured.",
                    done=True,
                )
                return body
            
            if self.valves.debug:
                logger.info(f"Model response data keys: {response_data.keys()}")

            # Extract content from response
            choices = response_data.get("choices", [])
            if not choices:
                logger.warning("No choices in model response, skipping tool selection")
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    "No response from model, skipping tool selection",
                    done=True,
                )
                return body
            
            content: str = choices[0].get("message", {}).get("content", "")
            
            if not content:
                logger.warning("Empty content in model response")
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    "Empty response from model, skipping tool selection",
                    done=True,
                )
                return body
            
            if self.valves.debug:
                logger.info(f"Model response content: {content}")

            selected_tool_ids: List[str] = self.parse_model_response(content)

            if self.valves.debug:
                logger.info(f"Parsed tool IDs: {selected_tool_ids}")

            # Step 6: Validate selected tools exist in available tools (case-insensitive matching)
            valid_ids_set = {t["id"].lower() for t in available_tools}
            valid_id_lookup = {t["id"].lower(): t["id"] for t in available_tools}  # Map lowercase to original

            if self.valves.force_custom_tools:
                forced_tools = [tool_id.strip() for tool_id in self.valves.force_custom_tools.split(",")]
                if self.valves.debug:
                    logger.info(f"Adding forced tool IDs: {forced_tools}")
                # Only add forced tools if they exist in available tools (case-insensitive)
                for tool_id in forced_tools:
                    if tool_id.lower() in valid_ids_set:
                        original_id = valid_id_lookup[tool_id.lower()]
                        if original_id not in selected_tool_ids:
                            selected_tool_ids.append(original_id)
                    else:
                        logger.warning(f"Force tool '{tool_id}' not found in available tools, ignoring")

            # Validate and normalize case for all selected tools
            selected_tool_ids = [
                valid_id_lookup.get(tid.lower(), tid)
                for tid in selected_tool_ids
                if tid.lower() in valid_ids_set
            ]

            # Step 7: Filter out similar tools
            if self.valves.similarity_threshold < 1.0:
                original_count = len(selected_tool_ids)
                selected_tool_ids = self.filter_similar_tools(
                    selected_tool_ids, available_tools, self.valves.similarity_threshold
                )
                if self.valves.debug and len(selected_tool_ids) < original_count:
                    logger.info(f"Filtered {original_count - len(selected_tool_ids)} similar tools")

            # Step 8: Update body with selected tools
            if selected_tool_ids:
                # Separate built-in features from custom tools
                builtin_features = [tid for tid in selected_tool_ids 
                                if tid in ["web_search", "image_generation", "code_interpreter"]]
                custom_tools = [tid for tid in selected_tool_ids 
                                if tid not in builtin_features]

                # Update features for built-in capabilities
                if builtin_features:
                    features: Dict[str, bool] = body.get("features", {})
                    for feature_id in builtin_features:
                        features[feature_id] = True
                    body["features"] = features

                # Update tool_ids for custom tools (use 'tools' key for Open WebUI compatibility)
                logger.info(f"Instructing use of tools: {custom_tools}")
                if custom_tools:
                    body["tools"] = custom_tools

                # Create status message
                feature_msg = f"{len(builtin_features)} feature{'s' if len(builtin_features) > 1 else ''}" if builtin_features else ""
                tool_msg = f"{len(custom_tools)} tool{'s' if len(custom_tools) > 1 else ''}" if custom_tools else ""
                parts = " and ".join([p for p in [feature_msg, tool_msg] if p])

                # Get the appropriate status message template
                msg_template = STATUS_MESSAGES.get("features_tools_msg",{}).get(self.s_key)
                if msg_template:
                    # Use safe_substitute to avoid KeyError if template doesn't have all placeholders
                    try:
                        status_desc = msg_template.safe_substitute(parts=parts, tool_ids=", ".join(selected_tool_ids))
                    except Exception as e:
                        if self.valves.debug:
                            logger.error(f"Error formatting status message: {e}")
                        status_desc = f"Selected {len(selected_tool_ids)} tool(s)"
                else:
                    status_desc = f"Selected {len(selected_tool_ids)} tool(s)"
                
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    status_desc,
                    done=True,
                )
            else:
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    STATUS_MESSAGES.get("no_tool_found",{}).get(self.s_key),
                    done=True,
                )

        except Exception as e:
            logger.exception(f"Error in AutoToolSelector.inlet: {e}")
            await self.emit_status(
                __event_emitter__,
                "error",
                f"Error selecting tools: {str(e)}",
                done=True,
            )

        return body