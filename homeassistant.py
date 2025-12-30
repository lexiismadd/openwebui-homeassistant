"""
title: Home Assistant Control
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 1.0.0
license: MIT
requirements: aiohttp
"""

import aiohttp
import asyncio
import fnmatch
import json
from typing import Callable, Any, Optional
from pydantic import BaseModel, Field


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
        self.entities_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 300  # Cache for 5 minutes

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
        # Parse configuration
        domains = [d.strip() for d in self.valves.DISCOVER_DOMAINS.split(",") if d.strip()]
        included = [p.strip() for p in self.valves.INCLUDED_ENTITIES.split(",") if p.strip()]
        excluded = [p.strip() for p in self.valves.EXCLUDED_ENTITIES.split(",") if p.strip()]

        filtered = []
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""

            # Check domain
            if domain not in domains:
                continue

            # Check excluded patterns first
            if excluded and self._matches_pattern(entity_id, excluded):
                continue

            # Check included patterns (if specified)
            if included and not self._matches_pattern(entity_id, included):
                continue

            filtered.append(entity)

        return filtered

    async def _get_entities(self, force_refresh: bool = False) -> list[dict]:
        """Get and cache filtered entities from Home Assistant."""
        import time
        current_time = time.time()

        # Use cache if valid and not forcing refresh
        if not force_refresh and self.entities_cache and (current_time - self.cache_timestamp) < self.cache_duration:
            return self.entities_cache

        # Fetch fresh data
        states = await self._make_request("GET", "states")
        filtered = self._filter_entities(states)

        # Update cache
        self.entities_cache = filtered
        self.cache_timestamp = current_time

        return filtered

    def _format_entity_info(self, entity: dict) -> str:
        """Format entity information for display."""
        entity_id = entity.get("entity_id", "Unknown")
        state = entity.get("state", "unknown")
        attributes = entity.get("attributes", {})
        friendly_name = attributes.get("friendly_name", entity_id)

        info = f"• {friendly_name} ({entity_id}): {state}"

        # Add relevant attributes based on domain
        domain = entity_id.split(".")[0]
        
        if domain == "light" and state == "on":
            if "brightness" in attributes:
                brightness_pct = round((attributes["brightness"] / 255) * 100)
                info += f" | Brightness: {brightness_pct}%"
            if "rgb_color" in attributes:
                info += f" | RGB: {attributes['rgb_color']}"
        elif domain == "climate":
            if "current_temperature" in attributes:
                info += f" | Current: {attributes['current_temperature']}°"
            if "temperature" in attributes:
                info += f" | Target: {attributes['temperature']}°"
            if "hvac_action" in attributes:
                info += f" | Action: {attributes['hvac_action']}"
        elif domain == "cover":
            if "current_position" in attributes:
                info += f" | Position: {attributes['current_position']}%"
        elif domain == "fan":
            if "percentage" in attributes:
                info += f" | Speed: {attributes['percentage']}%"
        elif domain in ["sensor", "binary_sensor"]:
            if "unit_of_measurement" in attributes:
                info += f" {attributes['unit_of_measurement']}"
            if "device_class" in attributes:
                info += f" | Type: {attributes['device_class']}"
        elif domain == "media_player":
            if "media_title" in attributes:
                info += f" | Playing: {attributes['media_title']}"
            if "volume_level" in attributes:
                volume_pct = round(attributes['volume_level'] * 100)
                info += f" | Volume: {volume_pct}%"

        return info

    async def discover_entities(
        self,
        refresh: bool = False,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Discover and list all configured Home Assistant entities.
        
        :param refresh: Force refresh the entity cache (default: False)
        :return: Formatted list of discovered entities organized by domain
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return "❌ Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Discovering Home Assistant entities...", "done": False},
                    }
                )

            entities = await self._get_entities(force_refresh=refresh)

            if not entities:
                return "No entities found matching the configured filters."

            # Organize by domain
            by_domain = {}
            for entity in entities:
                domain = entity["entity_id"].split(".")[0]
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(entity)

            # Format output
            result = f"📋 Discovered {len(entities)} entities across {len(by_domain)} domains:\n\n"
            
            for domain in sorted(by_domain.keys()):
                result += f"\n**{domain.upper()}** ({len(by_domain[domain])} entities):\n"
                for entity in sorted(by_domain[domain], key=lambda x: x.get("entity_id", "")):
                    result += self._format_entity_info(entity) + "\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Discovery complete", "done": True},
                    }
                )

            return result

        except Exception as e:
            error_msg = f"❌ Error discovering entities: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def get_entity_state(
        self,
        entity_id: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Get the current state and attributes of a specific Home Assistant entity.
        
        :param entity_id: The entity_id to query (e.g., 'light.living_room')
        :return: Detailed state information for the entity
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return "❌ Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Getting state for {entity_id}...", "done": False},
                    }
                )

            entity = await self._make_request("GET", f"states/{entity_id}")

            friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
            state = entity.get("state", "unknown")
            last_changed = entity.get("last_changed", "unknown")
            last_updated = entity.get("last_updated", "unknown")
            attributes = entity.get("attributes", {})

            result = f"**{friendly_name}** ({entity_id})\n\n"
            result += f"• **State**: {state}\n"
            result += f"• **Last Changed**: {last_changed}\n"
            result += f"• **Last Updated**: {last_updated}\n"

            if attributes:
                result += f"\n**Attributes**:\n"
                for key, value in sorted(attributes.items()):
                    if key != "friendly_name":  # Already displayed
                        result += f"  • {key}: {value}\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "State retrieved", "done": True},
                    }
                )

            return result

        except Exception as e:
            error_msg = f"❌ Error getting state for {entity_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        additional_data: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Call a Home Assistant service on one or more entities.
        
        :param domain: The domain of the service (e.g., 'light', 'switch', 'climate')
        :param service: The service to call (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature')
        :param entity_id: The entity_id or comma-separated list of entity_ids to target
        :param additional_data: Optional JSON string with additional service data (e.g., '{"brightness": 255}' for lights)
        :return: Confirmation of the service call
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return "❌ Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Calling {domain}.{service} on {entity_id}...", "done": False},
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
                    return f"❌ Invalid JSON in additional_data: {additional_data}"

            # Call the service
            result = await self._make_request(
                "POST",
                f"services/{domain}/{service}",
                data=service_data
            )

            success_msg = f"✅ Successfully called {domain}.{service} on {entity_id}"
            
            if additional_data:
                success_msg += f" with data: {additional_data}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Service call complete", "done": True},
                    }
                )

            # Invalidate cache to reflect changes
            self.entities_cache = {}

            return success_msg

        except Exception as e:
            error_msg = f"❌ Error calling {domain}.{service} on {entity_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def list_services(
        self,
        domain: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List available Home Assistant services, optionally filtered by domain.
        
        :param domain: Optional domain to filter services (e.g., 'light', 'switch'). If not provided, lists all services.
        :return: Formatted list of available services
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return "❌ Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Fetching available services...", "done": False},
                    }
                )

            services = await self._make_request("GET", "services")

            if domain:
                # Filter to specific domain
                if domain not in services:
                    return f"❌ Domain '{domain}' not found in Home Assistant services."
                services = {domain: services[domain]}

            result = "**Available Home Assistant Services**:\n\n"

            for service_domain, service_list in sorted(services.items()):
                result += f"\n**{service_domain.upper()}**:\n"
                for service_name, service_info in sorted(service_list.items()):
                    description = service_info.get("description", "No description")
                    result += f"  • **{service_name}**: {description}\n"
                    
                    # Add fields if available
                    fields = service_info.get("fields", {})
                    if fields:
                        result += f"    Fields: {', '.join(fields.keys())}\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Services listed", "done": True},
                    }
                )

            return result

        except Exception as e:
            error_msg = f"❌ Error listing services: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def validate_connection(self) -> dict:
        """
        Validate the Home Assistant connection when valves are saved.
        This is called automatically by Open WebUI when valve values are updated.
        """
        # Skip validation if not configured
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "status": "not_configured",
                "message": "Home Assistant URL and Token are not configured. Please configure them to enable the tool."
            }

        try:
            # Test connection by getting API status
            config = await self._make_request("GET", "config")
            
            # Get entity count
            entities = await self._get_entities(force_refresh=True)
            
            return {
                "status": "success",
                "message": f"✅ Connected to Home Assistant\n"
                          f"Location: {config.get('location_name', 'Unknown')}\n"
                          f"Version: {config.get('version', 'Unknown')}\n"
                          f"Discovered Entities: {len(entities)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Failed to connect to Home Assistant: {str(e)}\n"
                          f"Please check your HA_URL and HA_TOKEN configuration."
            }
