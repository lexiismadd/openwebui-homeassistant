# How the LLM Knows About Tool Functions

## Overview

When you install a tool in Open WebUI, the system automatically makes the tool's functions available to the LLM through **function calling** (also known as tool use). The LLM learns about available functions through their **docstrings**.

## How It Works

### 1. Function Discovery
When you add the Home Assistant tool to Open WebUI:
- Open WebUI scans the `Tools` class for all async methods
- Each method becomes an available "tool" that the LLM can call
- The method's docstring becomes the function's description

### 2. LLM Receives Function Definitions
When a user starts a conversation, Open WebUI sends the LLM:
```json
{
  "tools": [
    {
      "name": "get_entities_for_llm_context",
      "description": "[PRIMARY FUNCTION - CALL THIS FIRST] Get a comprehensive list...",
      "parameters": {
        "domain": "Optional domain filter...",
        "refresh": "Force cache refresh..."
      }
    },
    {
      "name": "call_service",
      "description": "Execute a Home Assistant service... PREREQUISITE: You MUST call get_entities_for_llm_context() first...",
      "parameters": {...}
    }
  ]
}
```

### 3. LLM Decides When to Use Functions
Based on the user's message and the function descriptions, the LLM decides:
- **Which function(s) to call**
- **In what order**
- **With what parameters**

### 4. Function Execution
When the LLM wants to use a function:
- LLM sends a function call request to Open WebUI
- Open WebUI executes the Python function
- Result is sent back to the LLM
- LLM uses the result to formulate its response

## Example Flow

**User:** "Turn on the toilet light"

**Step 1: LLM Analyzes**
```
LLM thinks: "User wants to control a Home Assistant light. 
I see a function called get_entities_for_llm_context that says 
'[PRIMARY FUNCTION - CALL THIS FIRST]' and mentions mapping 
user requests to entity_ids. I should call that first."
```

**Step 2: LLM Calls First Function**
```json
{
  "function": "get_entities_for_llm_context",
  "parameters": {
    "domain": "light"
  }
}
```

**Step 3: Open WebUI Executes**
```python
result = await tools.get_entities_for_llm_context(domain="light")
# Returns JSON with all light entities
```

**Step 4: LLM Receives Result**
```json
{
  "entities_by_domain": {
    "light": [
      {
        "entity_id": "light.toilet",
        "friendly_name": "Toilet Light",
        "state": "off"
      }
    ]
  }
}
```

**Step 5: LLM Analyzes Result**
```
LLM thinks: "Perfect! 'toilet light' maps to entity_id 'light.toilet'.
Now I see another function 'call_service' that says I must use 
exact entity_ids. I'll call that with the ID I just found."
```

**Step 6: LLM Calls Second Function**
```json
{
  "function": "call_service",
  "parameters": {
    "domain": "light",
    "service": "turn_on",
    "entity_id": "light.toilet"
  }
}
```

**Step 7: Success**
```python
result = await tools.call_service(
    domain="light", 
    service="turn_on", 
    entity_id="light.toilet"
)
# Returns: "✅ Successfully called light.turn_on on light.toilet"
```

**Step 8: LLM Responds to User**
```
"I've turned on the toilet light for you."
```

## Why Docstrings Matter

The quality of your docstrings directly affects how well the LLM uses your tool:

### ❌ Bad Docstring (Vague)
```python
async def call_service(self, domain, service, entity_id):
    """Call a service."""
```
**Problem:** LLM doesn't know:
- When to call this
- What parameters mean
- What prerequisites exist
- What format entity_id should be

### ✅ Good Docstring (Clear & Directive)
```python
async def call_service(self, domain, service, entity_id):
    """
    Execute a Home Assistant service on one or more entities.
    
    **PREREQUISITE: You MUST call get_entities_for_llm_context() first.**
    
    **Entity ID format:** Must be "domain.entity_name" (e.g., "light.kitchen")
    
    **Example workflow:**
    1. Call get_entities_for_llm_context(domain="light")
    2. Find entity_id from JSON response
    3. Call this function with exact entity_id
    
    :param domain: Service domain (e.g., 'light', 'switch')
    :param service: Service name (e.g., 'turn_on', 'turn_off')
    :param entity_id: Exact entity_id from get_entities_for_llm_context()
    """
```
**Benefits:** LLM knows:
- ✅ Must call get_entities_for_llm_context() first
- ✅ Entity_id must be in specific format
- ✅ Where to get valid entity_ids
- ✅ Example of proper workflow

## Best Practices for Tool Docstrings

### 1. Use Directive Language
```python
"""
**MUST call X first**
**NEVER do Y**
**ALWAYS use Z format**
"""
```

### 2. Include Examples
```python
"""
**Example:**
User: "Turn on kitchen light"
Step 1: get_entities_for_llm_context(domain="light")
Step 2: Find entity_id: "light.kitchen"
Step 3: call_service(domain="light", service="turn_on", entity_id="light.kitchen")
"""
```

### 3. Specify Prerequisites
```python
"""
**PREREQUISITE: You MUST call get_entities_for_llm_context() first.**
"""
```

### 4. Clarify When to Use
```python
"""
**Use THIS function when:**
- User asks to control a device
- You need to change state

**Use OTHER function when:**
- You need to get current state
"""
```

### 5. Define Formats Clearly
```python
"""
**Entity ID format:** Must be "domain.entity_name" (e.g., "light.kitchen", not "kitchen light")
**Multiple entities:** Use comma-separated list (e.g., "light.a,light.b")
"""
```

## Why This Approach Works

### Traditional Approach (String Matching)
```python
def resolve_entity(name):
    if "toilet" in name.lower():
        return "light.toilet"
    # Fails for: "loo light", "WC bulb", "bathroom main light"
```
**Problems:**
- Can't handle variations
- Hard to maintain
- Brittle to naming changes

### LLM Approach (Context + Reasoning)
```python
# LLM receives:
{
  "entity_id": "light.toilet",
  "friendly_name": "Toilet Light",
  "area": "bathroom"
}

# LLM can understand:
# "toilet light" → light.toilet ✅
# "loo light" → light.toilet ✅ (if it knows "loo" = toilet)
# "bathroom light" → light.toilet ✅ (area match)
# "WC illumination" → light.toilet ✅ (if it knows WC = toilet)
```
**Benefits:**
- Leverages LLM's language understanding
- Handles synonyms naturally
- Uses context clues (area, state, etc.)
- Adapts to unusual naming

## Debugging Tips

### Check Function is Available
In Open WebUI, when chatting with the LLM, it should show tool usage in the UI when a function is called.

### Verify Docstrings
The LLM literally reads your docstrings, so:
- **Be explicit**: Say "MUST" not "should"
- **Be directive**: Tell it what to do, not suggestions
- **Be clear**: Use examples and formatting
- **Be complete**: Include all prerequisites and formats

### Common Issues

**Issue:** LLM guesses entity_ids instead of calling get_entities_for_llm_context()
**Fix:** Make the prerequisite more prominent: "**PREREQUISITE: You MUST call...**"

**Issue:** LLM calls functions in wrong order
**Fix:** Add workflow examples showing the correct sequence

**Issue:** LLM doesn't understand parameters
**Fix:** Add concrete examples of valid parameter values

## Summary

The LLM learns about your tool functions through:
1. **Function names** (what to call)
2. **Docstrings** (when and how to call)
3. **Parameter types** (what data to pass)

The better your docstrings, the better the LLM will use your tool. Be explicit, directive, and include examples!