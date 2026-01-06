"""
title: Home Assistant Control
author: open-webui
author_url: https://github.com/lexiismadd
funding_url: https://github.com/open-webui
version: 3.2.0
license: MIT
requirements: aiohttp, loguru
"""

import difflib
import re
import aiohttp
import asyncio
import fnmatch
import json
from difflib import SequenceMatcher
from typing import Callable, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger


class Tools:
    class Valves(BaseModel):
        HA_URL: str = Field(
            default="",
            description="Home Assistant URL (e.g., http://homeassistant.local:8123)",
        )
        HA_TOKEN: str = Field(
            default="",
            description="Home Assistant Long-Lived Access Token",
        )
        DISCOVER_DOMAINS: str = Field(
            default="light,switch,climate,cover,fan,lock,media_player,sensor,binary_sensor,weather,camera,vacuum,scene,script,automation",
            description="Comma-separated list of Home Assistant domains to discover",
        )
        INCLUDED_ENTITIES: str = Field(
            default="",
            description="Comma-separated list of entity_id patterns to include (supports wildcards, e.g., 'light.living_room_*,switch.bedroom_*'). Leave empty to include all.",
        )
        EXCLUDED_ENTITIES: str = Field(
            default="",
            description="Comma-separated list of entity_id patterns to exclude (supports wildcards, e.g., 'sensor.*_battery,*_update')",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._entity_words_cache = None  # Cache for common words in entities
        logger.info("Home Assistant tool initialized")

    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[dict] = None
    ) -> dict:
        """Make an authenticated request to Home Assistant API."""
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            raise ValueError("Home Assistant URL and Token must be configured in valves")

        url = f"{self.valves.HA_URL.rstrip('/')}/api/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.valves.HA_TOKEN}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response.raise_for_status()
                        return await response.json()
            except aiohttp.ClientError as e:
                raise Exception(f"Home Assistant API request failed: {str(e)}")

    def _matches_pattern(self, entity_id: str, patterns: list[str]) -> bool:
        """Check if entity_id matches any of the patterns (supports wildcards)."""
        if not patterns:
            return False
        return any(fnmatch.fnmatch(entity_id, pattern.strip()) for pattern in patterns)

    def _filter_entities(self, entities: list[dict]) -> list[dict]:
        """Filter entities based on domain, included, and excluded patterns."""
        domains = [d.strip() for d in self.valves.DISCOVER_DOMAINS.split(",") if d.strip()]
        included = [p.strip() for p in self.valves.INCLUDED_ENTITIES.split(",") if p.strip()]
        excluded = [p.strip() for p in self.valves.EXCLUDED_ENTITIES.split(",") if p.strip()]

        filtered = []
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""

            if domain not in domains:
                continue
            if excluded and self._matches_pattern(entity_id, excluded):
                continue
            if included and not self._matches_pattern(entity_id, included):
                continue

            filtered.append(entity)

        return filtered

    def _format_entity_for_llm(self, entity: dict) -> dict:
        """Format entity information for LLM understanding."""
        entity_id = entity.get("entity_id", "")
        attributes = entity.get("attributes", {})
        state = entity.get("state", "unknown")
        
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        
        info = {
            "entity_id": entity_id,
            "domain": domain,
            "friendly_name": attributes.get("friendly_name", entity_id),
            "state": state,
        }
        
        # Add area/room if available
        if "area_name" in attributes:
            info["area"] = attributes["area_name"]
        
        # Add device class for sensors
        if "device_class" in attributes:
            info["device_class"] = attributes["device_class"]
            
        # Add unit of measurement for sensors
        if "unit_of_measurement" in attributes:
            info["unit"] = attributes["unit_of_measurement"]
        
        # Add relevant domain-specific attributes
        if domain == "light" and state == "on":
            if "brightness" in attributes:
                info["brightness_pct"] = round((attributes["brightness"] / 255) * 100)
        elif domain == "climate":
            if "current_temperature" in attributes:
                info["current_temp"] = attributes["current_temperature"]
            if "temperature" in attributes:
                info["target_temp"] = attributes["temperature"]
        elif domain == "cover":
            if "current_position" in attributes:
                info["position"] = attributes["current_position"]
                
        return info

    async def control_home_assistant(
        self,
        user_request: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        [PRIMARY FUNCTION] Control or query Home Assistant based on natural language request.
        
        This is the ONLY function you should call for Home Assistant interactions.
        It automatically handles the complete workflow:
        1. Refreshes entity cache from Home Assistant
        2. Intelligently filters to only relevant entities based on your request
        3. Returns focused entity list for you to analyze
        4. Tells you what to do next
        
        You MUST follow the 2-step workflow returned by this function.
        
        :param user_request: The user's natural language request (e.g., "turn on toilet light", "what's the temperature")
        :return: JSON object with instructions for next steps and filtered entity context
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "error": "NOT_CONFIGURED",
                "message": "Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."
            }

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Refreshing Home Assistant entities...", "done": False},
                    }
                )

            # ALWAYS refresh entities
            states = await self._make_request("GET", "states")
            entities = self._filter_entities(states)
            
            logger.info(f"Retrieved {len(entities)} total entities from Home Assistant")
            
            # Intelligently filter entities based on user request
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Identifying entity...", "done": False},
                    }
                )
            relevant_entities = self._filter_relevant_entities(entities, user_request)
            
            logger.info(f"Filtered to {len(relevant_entities)} relevant entities for request: {user_request}")
            
            # If still too many, provide summary instead of full list
            if len(relevant_entities) > 100:
                result = self._create_summary_response(relevant_entities, user_request)
            else:
                # Format for LLM
                formatted_entities = [self._format_entity_for_llm(e) for e in relevant_entities]
                
                # Group by domain
                by_domain = {}
                for entity_info in formatted_entities:
                    d = entity_info["domain"]
                    if d not in by_domain:
                        by_domain[d] = []
                    by_domain[d].append(entity_info)
                
                entity_context = {
                    "total_relevant_entities": len(formatted_entities),
                    "domains": list(by_domain.keys()),
                    "entities_by_domain": by_domain
                }
                
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Found {len(relevant_entities)} similar entities...", "done": False},
                        }
                    )
                
                result = {
                    "user_request": user_request,
                    "entity_context": entity_context,
                    "instructions": {
                        "step_1": "Look at entity_context.entities_by_domain to find entities matching the user's request",
                        "step_2": "Copy the EXACT entity_id string from the entity_context",
                        "step_3": "Call execute_action() with that EXACT entity_id - do NOT modify, shorten, or guess",
                        "critical_examples": {
                            "correct": "If you see 'entity_id': 'light.hallway_light', use exactly 'light.hallway_light'",
                            "incorrect": "Do NOT use 'light.hallway' or 'hallway_light' or any variation",
                            "correct2": "If you see 'entity_id': 'sensor.living_room_temperature_2', use exactly 'sensor.living_room_temperature_2'",
                            "incorrect2": "Do NOT use 'sensor.living_room' or 'sensor.living_room_temperature'"
                        }
                    }
                }
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Identifying matching entity from {len(relevant_entities)} entities...", "done": True},
                    }
                )
            
            return result

        except Exception as e:
            error_msg = f"Error getting Home Assistant context: {str(e)}"
            logger.error(error_msg)
            return {
                "error": "EXCEPTION",
                "message": error_msg
            }

    def _filter_relevant_entities(self, entities: list[dict], user_request: str, min_score: float = 0.75) -> list[dict]:
        """Filter entities to only those relevant to the user's request using fuzzy matching."""
        
        def preprocess(text):
            return text.replace(".", " ").replace("_", " ").lower().split()
        
        request_lower = user_request.lower()
        
        # Extract meaningful words from user request (remove common words)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "can", "turn", "check", "get", "set", "what",
                      "show", "me", "my", "on", "off", "to", "from", "in", "at", "of", "for",
                      "status", "state", "please", "thanks", "thank", "you"}
        
        request_words = [word for word in request_lower.split() 
                        if word not in stop_words and len(word) > 2]
        
        if not request_words:
            logger.warning(f"No meaningful words in request: {user_request}")
            # Return most common domains as fallback
            common_domains = ["light", "switch", "climate", "sensor"]
            filtered = [e for e in entities if e.get("entity_id", "").split(".")[0] in common_domains]
            return filtered[:50]
        
        logger.info(f"Request: {user_request}")
        logger.info(f"Entity list count: {len(entities)}")
        logger.info(f"Extracted words from request: {request_words}")
        
        # Score each entity based on relevance
        scored_entities = []

        for entity in entities:
            best_score = 0
            best_field = None
            parsed_entity = {
                "entity_id": entity.get("entity_id", "").lower(),
                "friendly_name": entity.get("attributes", {}).get("friendly_name", "").lower(),
                "area_name": entity.get("attributes", {}).get("area_name", "").lower(),
                "domain": entity.get("entity_id", "").split(".")[0] if "." in entity.get("entity_id", "") else ""
            }
            for field in ["entity_id", "friendly_name", "area_name", "domain"]:
                field_text = " ".join(preprocess(parsed_entity[field]))
                ratio = difflib.SequenceMatcher(None, request_lower, field_text).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_field = field
            scored_entities.append({
                "entity": entity,
                "best_field": best_field,
                "score": best_score
            })

        # Sort by score descending
        scored_entities.sort(key=lambda x: x["score"], reverse=True)
        
        # Extract entities
        relevant = [item["entity"] for item in scored_entities if item["score"] >= min_score]
        
        logger.info(f"Fuzzy matching found {len(relevant)} relevant entities (from {len(entities)} total)")
        
        # Log top matches for debugging
        if relevant and scored_entities:
            top_5 = scored_entities[:5] if len(scored_entities) >=5 else scored_entities
            logger.info(f"Top matches: {[(e['entity'].get('entity_id'), e['score']) for e in top_5]}")
        
        # Limit to reasonable number
        if len(relevant) > 200:
            relevant = relevant[:200]
            logger.info(f"Limited to 200 entities")
        
        # If no matches found with fuzzy search, return most common domains
        if not relevant:
            logger.warning(f"No fuzzy matches found for: {user_request}")
            common_domains = ["light", "switch", "climate", "sensor"]
            filtered = [e for e in entities if e.get("entity_id", "").split(".")[0] in common_domains]
            return filtered[:50]
        
        return relevant

    def _create_summary_response(self, entities: list[dict], user_request: str) -> dict:
        """Create a summary response when there are too many entities."""
        # Group by domain
        by_domain = {}
        for entity in entities:
            domain = entity.get("entity_id", "").split(".")[0]
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(entity)
        
        # Create summary
        summary = {
            "user_request": user_request,
            "total_entities": len(entities),
            "message": f"Found {len(entities)} potentially relevant entities. This is too many to list. Please be more specific.",
            "domains_found": {},
            "suggestion": "Try to be more specific in your request. Mention specific rooms, device names, or types."
        }
        
        # Add domain counts
        for domain, ents in by_domain.items():
            summary["domains_found"][domain] = {
                "count": len(ents),
                "example_entities": [
                    {
                        "entity_id": e.get("entity_id"),
                        "friendly_name": e.get("attributes", {}).get("friendly_name", "")
                    }
                    for e in ents[:3]  # Just show 3 examples per domain
                ]
            }
        
        return summary

    async def execute_action(
        self,
        action_type: str,
        entity_id: str,
        service: Optional[str] = None,
        friendly_name: Optional[str] = None,
        additional_data: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Execute an action on Home Assistant after getting context from control_home_assistant().
        
        ⚠️ CRITICAL: You must use the EXACT entity_id from control_home_assistant() response.
        Do NOT guess, truncate, or modify the entity_id in any way.
        
        Examples of CORRECT usage:
        - If context shows "light.hallway_light" → Use "light.hallway_light" ✅
        - If context shows "light.toilet" → Use "light.toilet" ✅
        
        Examples of INCORRECT usage:
        - Context shows "light.hallway_light" but you use "light.hallway" ❌
        - Context shows "sensor.living_room_temperature" but you use "sensor.living_room" ❌
        
        :param action_type: Either "get_state" or "call_service"
        :param entity_id: EXACT entity_id from control_home_assistant() - no modifications!
        :param service: Service name if action_type is "call_service" (e.g., "turn_on", "turn_off")
        :param friendly_name: Friendly name of the entity from control_home_assistant()
        :param additional_data: Optional JSON string with service parameters
        :return: JSON object with result of the action
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "error": "NOT_CONFIGURED",
                "message": "Home Assistant is not configured."
            }
        # Helper function to format responses
        def format_ha_response(raw_response: dict) -> str:
            """
            Clean and summarise Home Assistant tool output for display.
            
            Args:
                raw_response: The raw string or dict returned by the HA tool.
            
            Returns:
                Cleaned, concise string suitable for OpenWebUI.
            """
            entity_name = None
            
            # Handle dict response
            success = raw_response.get("success", True)
            domain = raw_response.get("domain")
            service = raw_response.get("service")
            entity_id = raw_response.get("entity_id")
            entity_name = raw_response.get("friendly_name", entity_id)
            message = raw_response.get("result", "")
            
            
            # Build concise message
            if service in ["turn_on", "turn_off"]:
                return f"{entity_name} turned {service.replace('_', ' ')}."
            elif service in ["set_temperature", "set_brightness"]:
                return f"{service.replace('_', ' ').capitalize()} executed for {entity_name}."
            elif message:
                # fallback: clean the HA message
                cleaned = re.sub(r"Successfully called \S+ on ", "", message)
                return f"{entity_name} updated." if not cleaned else cleaned
            
            # Fallback
            return f"Action {service} executed on {entity_name}."
            
        try:
            result = {
                "success": bool(),
                "action": action_type,
                "domain": str(),
                "service": str(),
                "friendly_name": str(),
                "entity_id": entity_id,
                "result": str(),
                "message": str()
            }
            
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"{str(action_type).replace('_',' ').capitalize()} {f'{service} ' if service else ''} on {entity_id}...", "done": False},
                    }
                )
            
            if action_type == "call_service":
                if not service:
                    return {
                        "error": "MISSING_PARAMETER",
                        "message": "service parameter is required when action_type is 'call_service'"
                    }
                
                # Get entity first to confirm it exists
                try:
                    entity = await self._make_request("GET", f"states/{entity_id}")
                except Exception as api_error:
                    # Entity not found - provide helpful suggestions
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise
                
                friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id) if not friendly_name else friendly_name
                state = entity.get("state", "unknown")
                attributes = entity.get("attributes", {})
                
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Attempting {service} for device {friendly_name if friendly_name else entity_id}...", "done": False},
                        }
                    )
                                    
                # Call service action
                domain = entity_id.split(".")[0] if "." in entity_id else ""
                
                if not domain:
                    return {
                        "error": "INVALID_ENTITY_ID",
                        "message": f"Invalid entity_id format: {entity_id}. Must be 'domain.entity_name'"
                    }

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Calling {service} on {domain} {friendly_name if friendly_name else entity_id}...", "done": False},
                        }
                    )

                # Prepare service data
                service_data = {"entity_id": entity_id}

                # Parse additional data if provided
                if additional_data:
                    try:
                        extra = json.loads(additional_data)
                        service_data.update(extra)
                    except json.JSONDecodeError:
                        return {
                            "error": "INVALID_JSON",
                            "message": f"Invalid JSON in additional_data: {additional_data}"
                        }

                # Call the service
                try:
                    await self._make_request(
                        "POST",
                        f"services/{domain}/{service}",
                        data=service_data
                    )
                except Exception as api_error:
                    # Entity not found in service call
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise
                
                result["success"] = True
                result["domain"] = domain
                result["service"] = service
                result["friendly_name"] = friendly_name
                result["entity_id"] = entity_id
                result["result"] = f"Successfully called {service} on {domain} {friendly_name if friendly_name else entity_id}"
                
                if additional_data:
                    result["parameters"] = json.loads(additional_data)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Successfully called {service} on {domain} {friendly_name if friendly_name else entity_id}", "done": True},
                        }
                    )

                logger.info(f"Called {domain}.{service} on {entity_id}")
                result["message"] = format_ha_response(result)
                
                action_type = "get_state"
                await asyncio.sleep(2)

            if action_type == "get_state":
                # Get state action
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Getting state for {friendly_name if friendly_name else entity_id}...", "done": False},
                        }
                    )

                try:
                    entity = await self._make_request("GET", f"states/{entity_id}")
                except Exception as api_error:
                    # Entity not found - provide helpful suggestions
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise
                
                friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
                state = entity.get("state", "unknown")
                attributes = entity.get("attributes", {})

                result["success"] = True
                result["domain"] = domain
                result["state"] = state
                result["friendly_name"] = friendly_name
                result["entity_id"] = entity_id
                result["last_changed"] = entity.get("last_changed", "unknown")
                result["last_updated"] = entity.get("last_updated", "unknown")
                result["attributes"] = attributes
                
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Retrieved latest state for {friendly_name if friendly_name else entity_id}", "done": True},
                        }
                    )

                logger.info(f"Retrieved state for {entity_id}")
                return result

            else:
                return {
                    "error": "INVALID_ACTION_TYPE",
                    "message": f"Invalid action_type: {action_type}. Must be 'get_state' or 'call_service'"
                }

        except Exception as e:
            error_msg = f"Error executing action: {str(e)}"
            logger.error(error_msg)
            return {
                "error": "EXCEPTION",
                "message": error_msg
            }

    async def _handle_entity_not_found(self, attempted_entity_id: str) -> dict:
        """Handle entity not found errors with fuzzy matching suggestions."""
        logger.warning(f"Entity not found: {attempted_entity_id}")
        
        # Extract domain and partial name
        domain = attempted_entity_id.split(".")[0] if "." in attempted_entity_id else ""
        attempted_name = attempted_entity_id.split(".")[1] if "." in attempted_entity_id else attempted_entity_id
        
        # Get fresh entities
        try:
            states = await self._make_request("GET", "states")
            entities = self._filter_entities(states)
            
            # Find similar entities using fuzzy matching
            candidates = []
            
            for entity in entities:
                entity_id = entity.get("entity_id", "")
                entity_domain = entity_id.split(".")[0] if "." in entity_id else ""
                entity_name = entity_id.split(".")[1] if "." in entity_id else ""
                friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
                
                # Only consider entities from the same domain
                if entity_domain != domain:
                    continue
                
                # Calculate similarity scores
                # Score 1: Compare entity names
                name_similarity = SequenceMatcher(None, attempted_name.lower(), entity_name.lower()).ratio()
                
                # Score 2: Compare with friendly name
                friendly_similarity = SequenceMatcher(None, attempted_entity_id.lower(), friendly_name.lower()).ratio()
                
                # Score 3: Check if attempted name is substring
                substring_bonus = 0.3 if attempted_name.lower() in entity_name.lower() else 0
                
                # Combined score (weighted)
                score = max(name_similarity * 0.5 + friendly_similarity * 0.3 + substring_bonus, 
                           name_similarity, 
                           friendly_similarity)
                
                if score > 0.4:  # Threshold for similarity
                    candidates.append({
                        "entity_id": entity_id,
                        "friendly_name": friendly_name,
                        "similarity_score": round(score, 2)
                    })
            
            # Sort by similarity score (highest first)
            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            if candidates:
                return {
                    "error": "ENTITY_NOT_FOUND",
                    "attempted_entity_id": attempted_entity_id,
                    "message": f"The entity_id '{attempted_entity_id}' does NOT exist in Home Assistant.",
                    "possible_matches": candidates[:10],  # Top 10 matches
                    "required_action": {
                        "step_1": "Call control_home_assistant() again with the same user request",
                        "step_2": "Find the correct entity_id from the response (likely one from possible_matches)",
                        "step_3": "Call execute_action() again with the EXACT correct entity_id"
                    },
                    "warning": "DO NOT use the attempted_entity_id - it is WRONG. DO NOT assume any state. You MUST call control_home_assistant() again."
                }
            else:
                return {
                    "error": "ENTITY_NOT_FOUND", 
                    "attempted_entity_id": attempted_entity_id,
                    "message": f"The entity_id '{attempted_entity_id}' does NOT exist in Home Assistant.",
                    "possible_matches": [],
                    "required_action": {
                        "step_1": "Call control_home_assistant() again with a more specific user request",
                        "step_2": "Look for entities in the '{domain}' domain",
                        "step_3": "Call execute_action() with the correct entity_id"
                    },
                    "warning": "DO NOT assume any state. You MUST call control_home_assistant() again."
                }
                
        except Exception as e:
            return {
                "error": "ENTITY_NOT_FOUND",
                "attempted_entity_id": attempted_entity_id,
                "message": f"The entity_id '{attempted_entity_id}' does NOT exist.",
                "search_error": str(e),
                "required_action": {
                    "step_1": "Call control_home_assistant() again to get the correct entity_id"
                },
                "warning": "DO NOT assume any state. You MUST call control_home_assistant() again."
            }

    async def validate_connection(self) -> dict:
        """
        Validate the Home Assistant connection when valves are saved.
        This is called automatically by Open WebUI when valve values are updated.
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "status": "not_configured",
                "message": "Home Assistant URL and Token are not configured. Please configure them to enable the tool."
            }

        try:
            config = await self._make_request("GET", "config")
            states = await self._make_request("GET", "states")
            filtered = self._filter_entities(states)
            
            logger.info(f"✅ Connected to Home Assistant - {len(filtered)} entities")
            return {
                "status": "success",
                "message": f"✅ Connected to Home Assistant\n"
                          f"Location: {config.get('location_name', 'Unknown')}\n"
                          f"Version: {config.get('version', 'Unknown')}\n"
                          f"Discovered Entities: {len(filtered)}"
            }
        except Exception as e:
            logger.error(f"❌ Failed to connect: {str(e)}")
            return {
                "status": "error",
                "message": f"❌ Failed to connect to Home Assistant: {str(e)}\n"
                          f"Please check your HA_URL and HA_TOKEN configuration."
            }