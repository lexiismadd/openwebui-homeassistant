# Home Assistant Control Tool - v3.0.0 (Enforced Workflow)

A Home Assistant control tool for Open WebUI that **enforces** the LLM to always refresh entity context before every action.

## 🚨 Major Change in v3.0

### The Problem with v2.0
In v2.0, we relied on docstrings to tell the LLM to call `get_entities_for_llm_context()` first. However, LLMs don't always follow documentation perfectly, leading to:
- ❌ LLM guessing entity_ids instead of looking them up
- ❌ LLM calling services with stale or incorrect entity_ids  
- ❌ Actions failing because entity_ids don't match

### The Solution in v3.0: Enforced Workflow

Version 3.0 **architecturally enforces** the correct workflow. There are now only **2 functions** the LLM can call:

1. **`control_home_assistant(user_request)`** - ALWAYS call this first
2. **`execute_action(...)`** - ONLY call this after step 1

The LLM **cannot bypass** this workflow because the functions are designed to work together.

## How It Works

### User Request Flow

**User:** "Turn on the toilet light"

### Step 1: LLM Calls control_home_assistant()
```python
control_home_assistant(user_request="turn on the toilet light")
```

**Returns:**
```json
{
  "user_request": "turn on the toilet light",
  "entity_context": {
    "total_entities": 25,
    "entities_by_domain": {
      "light": [
        {
          "entity_id": "light.toilet",
          "friendly_name": "Toilet Light",
          "state": "off",
          "area": "bathroom"
        },
        {
          "entity_id": "light.kitchen",
          "friendly_name": "Kitchen Main",
          "state": "on",
          "brightness_pct": 75
        }
      ]
    }
  },
  "instructions": {
    "step_1": "Analyze the entity_context to find the entity_id(s) that match the user's request",
    "step_2": "Call execute_action() with the exact entity_id(s) and action details",
    "important": "Use EXACT entity_ids from entity_context. Do not guess or modify them."
  }
}
```

### Step 2: LLM Analyzes Context
The LLM sees:
- User wants to turn on "toilet light"
- Entity context shows: `light.toilet` with friendly_name "Toilet Light" 
- Current state is "off"
- The LLM now knows the exact entity_id to use

### Step 3: LLM Calls execute_action()
```python
execute_action(
    action_type="call_service",
    entity_id="light.toilet",  # Exact ID from context
    service="turn_on"
)
```

**Returns:**
```
✅ Successfully called light.turn_on on light.toilet
```

## Key Features

### ✅ Guaranteed Fresh Data
- **Every** call to `control_home_assistant()` fetches fresh entity data
- No stale cache issues
- LLM always has current states

### ✅ Enforced Workflow  
- LLM cannot call services without getting context first
- The architecture makes it impossible to bypass
- No relying on LLM following instructions

### ✅ Flexible Name Matching
- User says "toilet light" → LLM finds `light.toilet`
- User says "bathroom light" → LLM can match by area
- User says "loo light" → LLM can infer from context
- LLM's natural language understanding does the matching

### ✅ Simpler API
- Only 2 functions instead of 7
- Clear, linear workflow
- Harder to misuse

## Installation

### Step 1: Create Long-Lived Access Token

1. Open Home Assistant
2. Click your profile (bottom left)
3. Scroll to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Name it "Open WebUI Integration"
6. **Copy the token** (you won't see it again!)

### Step 2: Install in Open WebUI

1. Log into Open WebUI as admin
2. Go to **Settings** → **Tools**
3. Click **"+"** to add new tool
4. Paste the contents of `homeassistant.py`
5. Click **Save**

### Step 3: Configure Valves

1. Find "Home Assistant Control" in tools list
2. Click settings icon (⚙️)
3. Set:
   - **HA_URL**: `http://homeassistant.local:8123` (or your HA URL)
   - **HA_TOKEN**: Your long-lived access token
   - **DISCOVER_DOMAINS**: Comma-separated domains (default is good)
   - **INCLUDED_ENTITIES**: Optional patterns to include
   - **EXCLUDED_ENTITIES**: Optional patterns to exclude
4. Click **Save** - it will validate the connection

## Configuration Examples

### Basic Configuration
```
HA_URL: http://homeassistant.local:8123
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate,sensor
INCLUDED_ENTITIES: (blank = all)
EXCLUDED_ENTITIES: (blank = none)
```

### Only Specific Rooms
```
HA_URL: http://192.168.1.50:8123
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate
INCLUDED_ENTITIES: light.living_room_*,light.bedroom_*,switch.living_room_*
EXCLUDED_ENTITIES: 
```

### Exclude Maintenance Entities
```
HA_URL: https://ha.mydomain.com
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate,sensor,binary_sensor
INCLUDED_ENTITIES: 
EXCLUDED_ENTITIES: sensor.*_battery,sensor.*_update,*_rssi
```

## Function Reference

### control_home_assistant(user_request)

**[PRIMARY FUNCTION - Always call this first]**

Refreshes entity cache and returns current context with instructions.

**Parameters:**
- `user_request` (str): Natural language request (e.g., "turn on toilet light", "what's the temperature")

**Returns:** JSON containing:
- `user_request`: Echo of the request
- `entity_context`: Complete entity information organized by domain
- `instructions`: Step-by-step guide for next action

**Always returns fresh data from Home Assistant.**

**Example:**
```python
# User: "Is the kitchen light on?"
control_home_assistant(user_request="Is the kitchen light on?")

# Returns entity context with current states
# LLM finds light.kitchen and sees state="on"
# LLM responds to user: "Yes, the kitchen light is on at 75% brightness"
```

### execute_action(action_type, entity_id, service=None, additional_data=None)

**[SECONDARY FUNCTION - Only call after control_home_assistant()]**

Execute an action on a specific entity using exact entity_id from context.

**Parameters:**
- `action_type` (str): Either "get_state" or "call_service"
- `entity_id` (str): **EXACT** entity_id from control_home_assistant() response
- `service` (str, optional): Service name if action_type="call_service" (e.g., "turn_on", "turn_off")
- `additional_data` (str, optional): JSON string with service parameters

**Returns:** Success/failure message or entity state details

**Examples:**
```python
# Get state
execute_action(
    action_type="get_state",
    entity_id="sensor.living_room_temperature"
)

# Turn on light
execute_action(
    action_type="call_service",
    entity_id="light.kitchen",
    service="turn_on"
)

# Turn on light with brightness
execute_action(
    action_type="call_service",
    entity_id="light.bedroom",
    service="turn_on",
    additional_data='{"brightness_pct": 50}'
)

# Set thermostat
execute_action(
    action_type="call_service",
    entity_id="climate.thermostat",
    service="set_temperature",
    additional_data='{"temperature": 72}'
)

# Control multiple lights
execute_action(
    action_type="call_service",
    entity_id="light.bedroom_main,light.bedroom_reading",
    service="turn_off"
)
```

## Usage Examples

### Simple Query

**User:** "Is the toilet light on?"

**LLM Workflow:**
```python
# Step 1: Get context
result1 = control_home_assistant(user_request="Is the toilet light on?")
# Parse JSON, find light.toilet, state="off"

# Step 2: No action needed, just inform user
# LLM responds: "No, the toilet light is currently off."
```

### Simple Control

**User:** "Turn on the kitchen light"

**LLM Workflow:**
```python
# Step 1: Get context
result1 = control_home_assistant(user_request="Turn on the kitchen light")
# Parse JSON, find light.kitchen

# Step 2: Execute action
result2 = execute_action(
    action_type="call_service",
    entity_id="light.kitchen",
    service="turn_on"
)
# Returns: "✅ Successfully called light.turn_on on light.kitchen"

# LLM responds: "I've turned on the kitchen light."
```

### Control with Parameters

**User:** "Set bedroom lights to 30%"

**LLM Workflow:**
```python
# Step 1: Get context
result1 = control_home_assistant(user_request="Set bedroom lights to 30%")
# Parse JSON, find light.bedroom_main, light.bedroom_reading

# Step 2: Execute action
result2 = execute_action(
    action_type="call_service",
    entity_id="light.bedroom_main,light.bedroom_reading",
    service="turn_on",
    additional_data='{"brightness_pct": 30}'
)

# LLM responds: "I've set the bedroom lights to 30% brightness."
```

### State Query

**User:** "What's the temperature in the living room?"

**LLM Workflow:**
```python
# Step 1: Get context
result1 = control_home_assistant(user_request="What's the temperature in the living room?")
# Parse JSON, find sensor.living_room_temperature, state="72.5"

# Step 2: Get detailed state (optional, can also just use state from step 1)
result2 = execute_action(
    action_type="get_state",
    entity_id="sensor.living_room_temperature"
)

# LLM responds: "The living room temperature is 72.5°F"
```

### Complex Request

**User:** "Turn off all bedroom lights except the reading light"

**LLM Workflow:**
```python
# Step 1: Get context
result1 = control_home_assistant(user_request="Turn off all bedroom lights except the reading light")
# Parse JSON, find: light.bedroom_main, light.bedroom_closet, light.bedroom_reading
# Exclude light.bedroom_reading per user request

# Step 2: Execute action
result2 = execute_action(
    action_type="call_service",
    entity_id="light.bedroom_main,light.bedroom_closet",
    service="turn_off"
)

# LLM responds: "I've turned off the bedroom lights, but left the reading light on."
```

## Why This Enforced Approach Works

### Previous Issues (v2.0)
```
User: "Turn on toilet light"

❌ LLM might:
- Call call_service(entity_id="toilet light") directly → Fails (wrong format)
- Call call_service(entity_id="light.toilet_light") → Fails (wrong guess)
- Forget to call get_entities_for_llm_context() first
```

### Current Approach (v3.0)
```
User: "Turn on toilet light"

✅ LLM must:
- Call control_home_assistant() → Gets fresh entity list
- See exact entity_id: "light.toilet"
- Call execute_action() with exact ID → Success
```

The architecture makes it **impossible** to skip the context refresh step.

## Common Scenarios

### Handling Ambiguity
If user request is ambiguous, the LLM can ask for clarification after seeing context:

```
User: "Turn on the light"
LLM: control_home_assistant(user_request="Turn on the light")
LLM sees: 5 different lights in different rooms
LLM responds: "I found several lights. Which one would you like me to turn on?"
  - Kitchen Light
  - Bedroom Light  
  - Bathroom Light
  - Toilet Light
  - Living Room Light
```

### Non-Standard Names
User says "loo light" instead of "toilet light":

```python
# Step 1: Context shows entity "light.toilet" with friendly_name "Toilet Light" in area "bathroom"
# Step 2: LLM infers "loo" = toilet = bathroom → Uses light.toilet
# Works! LLM's language understanding bridges the gap
```

### Multiple Actions
User wants multiple things:

```
User: "Turn on the bedroom lights and set them to 50%"

Step 1: control_home_assistant() → Gets all entities
Step 2: execute_action(entity_id="light.bedroom_main,light.bedroom_reading", 
                       service="turn_on",
                       additional_data='{"brightness_pct": 50}')
```

## Troubleshooting

### Connection Issues
**Error:** "Failed to connect to Home Assistant"
- Verify HA_URL is accessible from Open WebUI
- Check Home Assistant is running
- Ensure URL includes `http://` or `https://`
- Test URL in browser

**Error:** "401 Unauthorized"
- Token is invalid or expired
- Create new long-lived access token
- Ensure entire token was copied

### Entity Not Found
**Error:** "404 Not Found" when calling service
- Entity_id doesn't exist
- Check entity_id in Home Assistant
- Verify DISCOVER_DOMAINS includes the entity's domain
- Check INCLUDED_ENTITIES/EXCLUDED_ENTITIES filters

### Network (Docker)
If Open WebUI is in Docker:
- Use Docker host IP, not `localhost`
- Docker Desktop: `host.docker.internal`
- Docker Linux: Bridge network IP or host mode

## Security

- **Never share your access token**
- Use dedicated HA user with limited permissions
- Use HTTPS when possible
- Secure your Open WebUI instance

## Version Compatibility

- **Open WebUI**: v0.6.4+
- **Home Assistant**: 2025.12+
- **Python**: 3.11+

## Requirements

- `aiohttp`: HTTP library (auto-installed)
- `loguru`: Logging (auto-installed)

## Changelog

### Version 3.0.0 (Current) - Enforced Workflow
- **BREAKING**: Completely new API - only 2 functions
- **NEW**: `control_home_assistant()` - Primary function that ALWAYS refreshes cache
- **NEW**: `execute_action()` - Secondary function for execution
- **REMOVED**: All individual functions (get_entities_for_llm_context, call_service, get_entity_state, etc.)
- **ENFORCED**: LLM cannot bypass the refresh workflow
- **GUARANTEED**: Every action has fresh entity data
- **SIMPLIFIED**: Clear 2-step workflow impossible to misuse

**Migration from v2.0:**
- Replace `get_entities_for_llm_context()` → `control_home_assistant(user_request="...")`
- Replace `call_service()` → `execute_action(action_type="call_service", ...)`
- Replace `get_entity_state()` → `execute_action(action_type="get_state", ...)`
- Always start with `control_home_assistant()` - it's enforced now!

### Version 2.0.0
- LLM-first context approach
- Relied on docstrings for workflow
- Multiple functions exposed

### Version 1.x
- Automatic entity resolution in code
- Complex pattern matching logic

## License

MIT License

## Support

Issues or questions:
1. Check troubleshooting section
2. Verify Home Assistant logs
3. Check Open WebUI logs  
4. Ensure v2025.12+ of Home Assistant
5. Verify LLM is calling `control_home_assistant()` first