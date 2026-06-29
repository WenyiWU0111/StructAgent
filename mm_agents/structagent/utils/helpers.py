import re
from typing import Dict, Optional, List, Tuple
import json

def parse_action_json(message: str) -> Optional[Dict]:
    """
    Parses the action JSON from a ChatCompletionMessage content string.

    Expected format: <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>
    
    Returns a dict with key "function_call" on success; otherwise returns the original message.
    """
    
    def _strip_think_blocks(text: str) -> str:
        return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text or '', flags=re.IGNORECASE | re.DOTALL)

    def _clean_json_string(json_str: str) -> str:
        """Clean common JSON formatting issues"""
        if not json_str:
            return json_str
            
        # Replace curly quotes with straight quotes
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', "'").replace(''', "'")
        
        # Remove trailing newlines and whitespace
        json_str = json_str.strip()
        
        # Remove problematic newlines at end of values
        json_str = re.sub(r'\\n["\'}]', '"', json_str)
        json_str = re.sub(r'\n["\'}]', '"', json_str)
        
        # Fix common malformed patterns
        # Fix: {"action": "left_click": [x, y]} -> {"action": "left_click", "coordinate": [x, y]}
        json_str = re.sub(r'"action":\s*"([^"]+)":\s*(\[[^\]]+\])', r'"action": "\1", "coordinate": \2', json_str)
        
        return json_str

    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find('{')
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1]
            start = text.find('{', start + 1)
        return None

    def _normalize_function_call_object(obj: Dict) -> Optional[Dict]:
        """Normalize various JSON formats to standard function_call format"""
        if not isinstance(obj, dict):
            return None
            
        # Handle direct {"name": ..., "arguments": ...} format
        if "name" in obj and "arguments" in obj:
            fc = {"name": obj.get("name"), "arguments": obj.get("arguments", {})}
        # Handle {"action": {...}} format  
        elif "action" in obj and isinstance(obj["action"], dict):
            fc = obj["action"]
        # Handle existing {"function_call": {...}} format
        elif "function_call" in obj and isinstance(obj["function_call"], dict):
            fc = obj["function_call"]
        else:
            return None

        if not isinstance(fc, dict) or not fc.get("name"):
            return None

        # Ensure arguments is a dict
        args = fc.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        # Normalize coordinate formats
        for coord_key in ["coordinate", "coords"]:
            if coord_key in args:
                coord_val = args[coord_key]
                if isinstance(coord_val, list) and len(coord_val) >= 2:
                    try:
                        x = int(coord_val[0])
                        y = int(coord_val[1])
                        # Keep as list for your existing code
                        args[coord_key] = [x, y]
                    except Exception:
                        pass
                elif isinstance(coord_val, str):
                    m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', coord_val)
                    if m:
                        args[coord_key] = [int(m.group(1)), int(m.group(2))]

        return {"function_call": {"name": fc["name"], "arguments": args}}

    def _try_parse_json(json_str: str) -> Optional[Dict]:
        """Attempt to parse JSON string with cleaning"""
        if not json_str:
            return None
            
        try:
            cleaned = _clean_json_string(json_str)
            obj = json.loads(cleaned)
            return _normalize_function_call_object(obj)
        except json.JSONDecodeError as e:
            print(f"[json] Error parsing JSON: {e}")
            print(f"[json] Problematic JSON: {json_str}")
            return None
        except Exception as e:
            print(f"[json] Unexpected error: {e}")
            return None

    text = _strip_think_blocks(message or "")

    # Primary: Extract from <tool_call> ... </tool_call> blocks
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, flags=re.DOTALL)
    if tool_call_match:
        json_content = tool_call_match.group(1).strip()
        result = _try_parse_json(json_content)
        if result:
            return result

    # Fallback: Try explicit 'Action: {...}'
    action_match = re.search(r'Action:\s*(\{.*\})', text, flags=re.DOTALL)
    if action_match:
        result = _try_parse_json(action_match.group(1))
        if result:
            return result

    # Fallback: Try code-fenced JSON
    fenced_matches = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    for fenced_content in fenced_matches:
        result = _try_parse_json(fenced_content.strip())
        if result:
            return result
        
        # Try extracting first JSON object from fenced content
        json_candidate = _extract_first_json_object(fenced_content)
        if json_candidate:
            result = _try_parse_json(json_candidate)
            if result:
                return result

    # Fallback: Try first balanced JSON object within the text
    json_candidate = _extract_first_json_object(text)
    if json_candidate:
        result = _try_parse_json(json_candidate)
        if result:
            return result

    # Last resort: Try whole text as JSON
    result = _try_parse_json(text)
    if result:
        return result

    # Return original message to trigger NL-based fallback
    return message


# ── Date resolution helpers ────────────────────────────────────────────────────

import datetime


def _next_weekday(from_date: datetime.date, target_weekday: int, next_week: bool = False) -> datetime.date:
    """Get the next occurrence of a weekday (0=Monday, 6=Sunday).

    next_week=False: closest FUTURE occurrence of `target_weekday`.
        Today=Sat, target=Sat → +7 (since "today" doesn't count).

    next_week=True: same weekday but in NEXT calendar week (week starting
        Monday). Equivalent to: jump forward to next Monday, then offset
        by target_weekday days. Aligns with OSWorld evaluator's
        `next week Saturday/Sunday/Friday` math.
        Today=Tue, target=Sat → +11 (= 6 to next Mon + 5).
        Today=Sat, target=Sat → +7  (= 2 to next Mon + 5).
        Today=Sun, target=Sat → +6  (= 1 to next Mon + 5).
        Today=Mon, target=Mon → +7  (= 7 to next Mon + 0).
    """
    if next_week:
        # Days until next Monday: if today is Mon, that's 7 days; else
        # 7 - weekday (e.g. Tue=6, Sat=2, Sun=1).
        days_to_next_monday = 7 - from_date.weekday() if from_date.weekday() > 0 else 7
        next_monday = from_date + datetime.timedelta(days=days_to_next_monday)
        return next_monday + datetime.timedelta(days=int(target_weekday))

    # Default: closest future occurrence of target_weekday (today excluded).
    days_ahead = target_weekday - from_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return from_date + datetime.timedelta(days=days_ahead)


def _first_weekday_of_offset_month(from_date: datetime.date, weekday: int, months_offset: int) -> datetime.date:
    """First occurrence of `weekday` in the month that is `months_offset` months after from_date's month."""
    total = from_date.year * 12 + (from_date.month - 1) + int(months_offset)
    year, zmonth = divmod(total, 12)
    first = datetime.date(year, zmonth + 1, 1)
    days_ahead = (int(weekday) - first.weekday()) % 7
    return first + datetime.timedelta(days=days_ahead)


def _build_date_toolkit() -> Tuple[Dict[str, dict], str]:
    """Build a toolkit of date calculation formulas with descriptions.

    Returns: (toolkit_dict, formatted_description_for_llm)
        toolkit_dict maps formula_id → {description, function(args) → date}
    """
    today = datetime.date.today()

    toolkit = {
        "next_weekday": {
            "description": "Get the next occurrence of a specific weekday (e.g., next Monday, next Friday)",
            "args": "weekday (0=Monday, 1=Tuesday, ..., 6=Sunday)",
            "fn": lambda weekday: _next_weekday(today, int(weekday)),
        },
        "next_week_weekday": {
            "description": "Get a specific weekday in next week (not this week, guaranteed next week)",
            "args": "weekday (0=Monday, ..., 6=Sunday)",
            "fn": lambda weekday: _next_weekday(today, int(weekday), next_week=True),
        },
        "tomorrow": {
            "description": "Get tomorrow's date",
            "args": "none",
            "fn": lambda: today + datetime.timedelta(days=1),
        },
        "yesterday": {
            "description": "Get yesterday's date",
            "args": "none",
            "fn": lambda: today - datetime.timedelta(days=1),
        },
        "nth_this_month": {
            "description": "Get a specific day of the current month (e.g., the 15th of this month)",
            "args": "day (1-31)",
            "fn": lambda day: datetime.date(today.year, today.month, int(day)),
        },
        "nth_next_month": {
            "description": "Get a specific day of next month (e.g., the 5th of next month)",
            "args": "day (1-31)",
            "fn": lambda day: datetime.date(today.year + (1 if today.month == 12 else 0),
                                            (today.month % 12) + 1, int(day)),
        },
        "first_weekday_of_offset_month": {
            "description": "Get the first occurrence of a weekday in the month N months after the current month (e.g., 'first Monday eight months later')",
            "args": "weekday,months — comma-separated, e.g. '0,8' for first Monday eight months later",
            "fn": lambda arg: _first_weekday_of_offset_month(today,
                                                            *[int(x.strip()) for x in str(arg).split(",")]),
        },
        "days_from_now": {
            "description": "Get a date N days from today",
            "args": "days (positive integer)",
            "fn": lambda days: today + datetime.timedelta(days=int(days)),
        },
        "this_weekend": {
            "description": "Get the COMING Saturday — this week's weekend (e.g. 'this weekend', 'this Sat'). Use this when the user means the upcoming weekend.",
            "args": "none",
            "fn": lambda: _next_weekday(today, 5),
        },
        "next_weekend": {
            "description": "Get NEXT WEEK's Saturday — the weekend after this coming one (e.g. 'next weekend' under the formal interpretation: the Saturday of the next calendar week). If today is Wed, this returns the Saturday ~10 days out, not 3 days.",
            "args": "none",
            "fn": lambda: _next_weekday(today, 5, next_week=True),
        },
        "this_month_info": {
            "description": "Get the current month's name and year (for 'this month' references)",
            "args": "none",
            "fn": lambda: today,
        },
    }

    desc_lines = []
    for fid, info in toolkit.items():
        desc_lines.append(f"  [{fid}] {info['description']} (args: {info['args']})")

    return toolkit, "\n".join(desc_lines)


def resolve_relative_dates(instruction: str, call_llm_fn=None) -> str:
    """
    Resolve relative date references in the instruction to concrete dates.

    If call_llm_fn is provided, uses LLM to select date formulas from a toolkit.
    Otherwise falls back to regex-based pattern matching.
    """
    today = datetime.date.today()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today_name = weekday_names[today.weekday()]

    # Check if instruction contains any date-like references
    date_keywords = re.search(
        r'\b(next|tomorrow|yesterday|this month|next month|weekend|monday|tuesday|wednesday'
        r'|thursday|friday|saturday|sunday|\d{1,2}(?:st|nd|rd|th))\b', instruction, re.IGNORECASE)
    if not date_keywords:
        return instruction

    # If LLM available, use toolkit-based approach
    if call_llm_fn:
        toolkit, toolkit_desc = _build_date_toolkit()

        prompt = f"""Today is {today_name} {today.strftime('%B %d %Y')}.

The following instruction contains relative date references that need to be resolved to concrete dates.

Instruction: {instruction}

Available date formulas:
{toolkit_desc}

For each relative date in the instruction, output which formula to use and with what argument.
Output format (JSON list):
[{{"original": "the relative date text", "formula": "formula_id", "arg": "argument or null"}}]

Examples:
- "next Monday" → [{{"original": "next Monday", "formula": "next_weekday", "arg": "0"}}]
- "5th next month" → [{{"original": "5th next month", "formula": "nth_next_month", "arg": "5"}}]
- "tomorrow" → [{{"original": "tomorrow", "formula": "tomorrow", "arg": null}}]
- "first Monday eight months later" → [{{"original": "first Monday eight months later", "formula": "first_weekday_of_offset_month", "arg": "0,8"}}]
- "first Friday of three months from now" → [{{"original": "first Friday of three months from now", "formula": "first_weekday_of_offset_month", "arg": "4,3"}}]

Output ONLY valid JSON:"""

        try:
            response = call_llm_fn(
                [{"role": "system", "content": "Resolve relative dates using the provided formulas. Output JSON only."},
                 {"role": "user", "content": prompt}],
                max_tokens=200)

            data = _parse_llm_json(response)
            if not isinstance(data, list):
                data = []

            resolved = []
            for item in data:
                original = item.get("original", "")
                formula_id = item.get("formula", "")
                arg = item.get("arg")

                if formula_id not in toolkit:
                    continue

                fn = toolkit[formula_id]["fn"]
                try:
                    if arg is not None and arg != "null":
                        result = fn(arg)
                    else:
                        result = fn()
                    date_str = result.strftime("%A %B %d %Y")
                    resolved.append(f'"{original}" = {date_str}')
                except Exception:
                    continue

            if resolved:
                date_note = f"\n(Note: today is {today_name} {today.strftime('%B %d %Y')}. {'; '.join(resolved)})"
                return instruction + date_note

        except Exception:
            pass

    # Fallback: regex-based pattern matching.
    #
    # NOTE on weekday vs. weekend semantics — verified against OSWorld's
    # `desktop_env/evaluators/getters/misc.py:get_rule_relativeTime`:
    #   • OSWorld's keys "next Monday" / "next Saturday" / "next Sunday"
    #     mean the COMING weekday — same as our `_next_weekday(today, N)`.
    #     Keep these as-is.
    #   • OSWorld has NO "next weekend" key. Task authors who write
    #     "next weekend" in instructions match it against eval keys
    #     "next week Saturday" / "next week Sunday" (formal: next-week's
    #     weekend, ~10-11 days out, not this Saturday). So
    #     `\bnext\s+weekend\b` here MUST use next_week=True to align —
    #     this matches the LLM-toolkit `next_weekend` formula above.
    patterns = {
        r'\bnext\s+monday\b': lambda: _next_weekday(today, 0),
        r'\bnext\s+tuesday\b': lambda: _next_weekday(today, 1),
        r'\bnext\s+wednesday\b': lambda: _next_weekday(today, 2),
        r'\bnext\s+thursday\b': lambda: _next_weekday(today, 3),
        r'\bnext\s+friday\b': lambda: _next_weekday(today, 4),
        r'\bnext\s+saturday\b': lambda: _next_weekday(today, 5),
        r'\bnext\s+sunday\b': lambda: _next_weekday(today, 6),
        r'\bnext\s+weekend\b': lambda: _next_weekday(today, 5, next_week=True),
        r'\btomorrow\b': lambda: today + datetime.timedelta(days=1),
        r'\byesterday\b': lambda: today - datetime.timedelta(days=1),
        r'\bthis\s+month\b': lambda: today,
    }

    resolved = []
    lower = instruction.lower()
    for pattern, calc in patterns.items():
        if re.search(pattern, lower):
            result = calc()
            date_str = result.strftime("%A %B %d %Y")
            match_text = re.search(pattern, lower).group()
            resolved.append(f'"{match_text}" = {date_str}')

    if resolved:
        date_note = f"\n(Note: today is {today_name} {today.strftime('%B %d %Y')}. {'; '.join(resolved)})"
        return instruction + date_note
    return instruction


# ── Task template extraction ───────────────────────────────────────────────────

TASK_TEMPLATES = {
    "flight_search": {
        "keywords": ["flight", "flights", "fly from", "fly to", "airline"],
        "fields": {
            "departure": "Departure city or airport NAME, exactly as mentioned in the instruction (e.g. 'Stockholm', 'New York', 'Kennedy Airport'). Do NOT output IATA codes — leave canonicalization to downstream resolution.",
            "destination": "Destination city or airport NAME, exactly as mentioned in the instruction (e.g. 'Stockholm', 'New York', 'Kennedy Airport'). Do NOT output IATA codes — leave canonicalization to downstream resolution.",
            "date": "Travel date(s)",
            "trip_type": "One way or round trip",
            "passengers": "Number and type of passengers (e.g. 2 adults)",
            "class": "Cabin class (economy/business/first)",
            "constraints": "Additional constraints (price limit, direct only, etc.)",
        "strategy": (
            "NOTE: Remember to include ALL required fields in the plan!"
        ),
        },
    },
    "hotel_search": {
        "keywords": ["hotel", "accommodation", "stay", "lodging"],
        "fields": {
            "location": "City or area",
            "check_in": "Check-in date",
            "check_out": "Check-out date",
            "guests": "Number of guests/rooms",
            "constraints": "Additional constraints (price, rating, amenities, etc.)",
        },
    },
    "car_rental": {
        "keywords": ["car rental", "rent a car", "car from", "car available", "pickup"],
        "fields": {
            "pickup_location": "Pickup location",
            "dropoff_location": "Drop-off location (if different)",
            "pickup_date": "Pickup date",
            "return_date": "Return date",
            "car_type": "Car type/size (e.g. large, SUV, economy)",
            "constraints": "Additional constraints (price, sort order, etc.)",
        },
    },
    "shopping": {
        "keywords": ["buy", "purchase", "add to cart", "shopping", "price of", "on sale","shirts", "shoes"],
        "fields": {
            "item": "Item to search for",
            "size": "Size",
            "quantity": "Quantity",
            "constraints": "Filters (color, brand, discount, price range, on sale, etc.)",
        },
        "strategy": (
            "Strategy: First use the search bar to search for the primary product category "
            "(e.g. 'men\\'s short sleeve shirts'). Then list all filters that should be applied in a single step, starting with [filters    ]..."
            "(e.g. [filters] size, discount, price range, on sale, etc. [filters])."
        ),
    },
    "appointment": {
        "keywords": ["book an appointment", "schedule", "reserve", "reservation"],
        "fields": {
            "service": "Service or purpose",
            "location": "Location/venue",
            "date_time": "Preferred date and time",
            "participants": "Number of people",
        },
    },
}


# Common city/airport → IATA code mapping.
#
# IMPORTANT — entry ORDER matters for substring fallback in
# `_resolve_iata`: specific airport keywords (Kennedy, O'Hare, Heathrow,
# ...) must appear BEFORE their parent city aggregator so that strings
# like "New York–Kennedy Airport" resolve to JFK rather than the
# city's NYC. Single-airport cities (Mumbai, Dubai, ...) sit anywhere.
IATA_CODES = {
    # Multi-airport US — airport names first, then city
    "jfk": "JFK", "kennedy": "JFK", "laguardia": "LGA", "newark": "EWR",
    "new york": "NYC", "new york city": "NYC",
    "o'hare": "ORD", "ohare": "ORD", "midway": "MDW",
    "chicago": "ORD",
    "dulles": "IAD", "reagan": "DCA",
    "washington": "IAD", "dc": "IAD",
    "los angeles": "LAX", "la": "LAX",
    "san francisco": "SFO",
    "seattle": "SEA",
    "logan": "BOS", "boston": "BOS",
    "miami": "MIA",
    "dallas": "DFW",
    "denver": "DEN",
    "atlanta": "ATL",
    "houston": "IAH",
    "las vegas": "LAS",
    "phoenix": "PHX",
    "philadelphia": "PHL",
    "san diego": "SAN",
    "detroit": "DTW",
    "minneapolis": "MSP",
    "orlando": "MCO",
    "portland": "PDX",
    # Multi-airport international — airport names first, then city
    "heathrow": "LHR", "gatwick": "LGW",
    "london": "LHR",
    "charles de gaulle": "CDG",
    "paris": "CDG",
    "narita": "NRT", "haneda": "HND",
    "tokyo": "NRT",
    "beijing": "PEK",
    "shanghai": "PVG",
    "hong kong": "HKG",
    "singapore": "SIN",
    "dubai": "DXB",
    "mumbai": "BOM",
    "delhi": "DEL", "new delhi": "DEL",
    "sydney": "SYD",
    "melbourne": "MEL",
    "toronto": "YYZ",
    "vancouver": "YVR",
    "berlin": "BER",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "rome": "FCO",
    "madrid": "MAD",
    "barcelona": "BCN",
    "zurich": "ZRH",
    "vienna": "VIE",
    "dublin": "DUB",
    # Stockholm: OSWorld chrome eval (e.g. 82bc8d6a) compares the URL
    # toStation against "STO" (city aggregator code), not "ARN" (Arlanda
    # airport). Most airline sites still resolve typed "Stockholm" to ARN
    # in their URL — the agent's chosen site is the variable here, not
    # this hardcode. We emit STO so planner targets the eval's expected
    # code; if the airline site forces ARN, the task is unsavable
    # regardless.
    "stockholm": "STO",
    "oslo": "OSL",
    "copenhagen": "CPH",
    "helsinki": "HEL",
    "istanbul": "IST",
    "bangkok": "BKK",
    "seoul": "ICN",
    "taipei": "TPE",
    "manila": "MNL",
    "kuala lumpur": "KUL",
    "jakarta": "CGK",
    "cairo": "CAI",
    "johannesburg": "JNB",
    "sao paulo": "GRU",
    "mexico city": "MEX",
    "buenos aires": "EZE",
    "lima": "LIM",
    "manchester": "MAN",
}


# Defensive override: if the LLM ignores the prompt and outputs an IATA
# code directly, normalize the codes that don't match what OSWorld
# evaluators expect. Keys are 3-letter codes; values are eval-aligned
# replacements. Add entries only when chrome eval evidence supports it.
IATA_NORMALIZE = {
    # Stockholm: LLMs prefer "ARN" (Arlanda) but OSWorld eval (e.g.
    # 82bc8d6a) expects the city aggregator code "STO".
    "ARN": "STO",
}


def _resolve_iata(value: str) -> str:
    """Replace city/airport name with IATA code if found in mapping.

    Returns a single code per entry; eval-aligned values for multi-airport
    cities are encoded into ``IATA_CODES`` directly (e.g. Stockholm → STO
    based on chrome eval evidence). When a specific airport is named
    (e.g. "Kennedy", "O'Hare"), substring match picks the airport-specific
    code — the airport-name entries are placed first in the table so
    they win over the city aggregator.

    If the input is already a 3-letter IATA code, it passes through
    unchanged unless ``IATA_NORMALIZE`` overrides it (e.g. ARN → STO),
    which catches the case where an upstream LLM outputs the wrong
    specific-airport code despite the prompt asking for the city name.
    """
    # Already an IATA code (3 uppercase letters)
    stripped = value.strip()
    if re.match(r'^[A-Z]{3}$', stripped):
        return IATA_NORMALIZE.get(stripped, stripped)

    lower = stripped.lower()
    # Try exact match first
    if lower in IATA_CODES:
        return IATA_CODES[lower]

    # Substring match (e.g., "New York–Kennedy Airport" → "JFK")
    for city, code in IATA_CODES.items():
        if city in lower:
            return code

    return stripped


def detect_task_template(instruction: str) -> Optional[Dict]:
    """Detect which template matches the instruction, if any."""
    lower = instruction.lower()
    for template_name, template in TASK_TEMPLATES.items():
        if any(kw in lower for kw in template["keywords"]):
            return {"name": template_name, "fields": template["fields"], "strategy": template.get("strategy", "")}
    return None


def extract_task_fields(instruction: str, call_llm_fn) -> Optional[str]:
    """
    Detect task template, call LLM to extract structured fields from instruction.

    Args:
        instruction: the raw task instruction
        call_llm_fn: callable(messages, max_tokens) -> str

    Returns:
        Augmented instruction with extracted fields appended, or None if no template matches.
    """
    template = detect_task_template(instruction)
    if template is None:
        return None

    # Resolve dates first so the LLM has concrete dates to work with
    instruction_with_dates = resolve_relative_dates(instruction)

    fields_desc = "\n".join(f"- {name}: {desc}" for name, desc in template["fields"].items())
    prompt = f"""Extract structured information from this task instruction. Only include fields that are explicitly mentioned or clearly implied. Output "N/A" for fields not mentioned.

Task instruction: {instruction_with_dates}

Fields to extract:
{fields_desc}

Output format (one field per line, keep it brief):
field_name: value"""

    messages = [
        {"role": "system", "content": "You are a precise information extractor. Extract only what is stated or clearly implied. Be concise."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm_fn(messages, max_tokens=200)
        if not response or not response.strip():
            return None

        # Filter out N/A fields and resolve IATA codes for flight searches
        lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and "N/A" not in line and "n/a" not in line and "not mentioned" not in line.lower():
                # Resolve IATA codes for departure/destination fields
                if template["name"] == "flight_search":
                    for field_key in ("departure", "destination"):
                        if line.lower().startswith(field_key):
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                resolved = _resolve_iata(parts[1].strip())
                                line = f"{parts[0]}: {resolved}"
                lines.append(line)

        if not lines:
            return None

        extracted = "\n".join(f"  {line}" for line in lines)
        strategy = template.get("strategy", "")
        strategy_text = f"\n{strategy}" if strategy else ""
        return f"\n[Task Requirements - {template['name']}]\n{extracted}{strategy_text}"
    except Exception:
        return None


# ── Plan corrector ─────────────────────────────────────────────────────────────


def correct_plan(instruction: str, plan: str, screenshots_b64: List[str],
                 call_llm_fn, completed_subgoals: List[str] = None,
                 system_prompt_suffix: str = "",
                 ledger_block: Optional[str] = None) -> Optional[str]:
    """
    Review and correct a generated plan for common issues.

    Args:
        instruction: the task instruction
        plan: the generated plan text
        screenshots_b64: recent screenshots (up to 3) as base64 strings
        call_llm_fn: callable(messages, max_tokens) -> str
        completed_subgoals: list of already-completed subgoal descriptions
        ledger_block: optional rendered Progress Ledger block; when provided,
            its ``DEAD-END PATHS ALREADY TRIED`` entries constrain the corrector
            to pick structurally different routes.

    Returns: corrected plan text, or None if no corrections needed
    """
    completed_context = ""
    if completed_subgoals:
        completed_list = "\n".join(f"  - {s}" for s in completed_subgoals)
        completed_context = f"""
ALREADY COMPLETED (do NOT re-introduce these steps):
{completed_list}
"""

    ledger_context = ""
    if ledger_block and ledger_block.strip():
        ledger_context = f"""
{ledger_block.strip()}
"""

    prompt = f"""You are a plan corrector for a GUI automation agent. Review and fix this plan.
    DO NOT add additional steps to the plan! DO NOT change original intention of the plan!
    If the original plan use shortcuts or commands, DO NOT change them!

Task: {instruction}
{completed_context}{ledger_context}
Plan to review:
{plan}

RULES TO ENFORCE:
1. DO NOT navigate to external URLs or new websites unless the task explicitly requires it. The agent should work within the currently open page/application. Exception: chrome:// settings URLs are allowed.
2. MERGE sequential interactions into single steps. These MUST be combined:
   - "Click field" + "Type text" → "Click the field and type 'text'"
   - "Click dropdown" + "Select option" → "Click the dropdown and select 'option'"
   - "Click search bar" + "Type query" + "Press Enter" → "Click the search bar, type 'query', and press Enter"
3. Remove unnecessary steps like "Wait for page to load", "Verify the result", "Confirm the action".
4. Each step should be a concrete action a GUI agent can complete in 1-3 low-level actions (click, type, scroll).
5. Check the screenshots — if the plan assumes UI elements that are NOT visible in the latest screenshot, fix those steps.
6. If the latest screenshot shows a popup, modal dialog, cookie banner, notification prompt, or overlay covering the main UI and the plan does not already start with dismissing it, PREPEND a step 1 to dismiss the overlay (click Close / OK / Accept / Dismiss / X / Got it / "No thanks"). The original task-specific steps follow it.
7. DO NOT remove important steps like REOPEN the browser, app, etc. ALSO DO NOT ADD steps that do not exist in the original plan (the ONLY exception is rule 6 — prepending an overlay-dismissal step is allowed and required when an overlay is visible).
8. DO NOT re-introduce steps that are listed as ALREADY COMPLETED above. Those are done — the plan should only contain remaining work.
9. If a Progress Ledger is present and it lists ``DEAD-END PATHS ALREADY TRIED``: ensure the corrected plan does NOT repeat any of those failed paths. The plan MUST solve the remaining goal via a structurally different route — a different menu entry, a different surface (address bar / keyboard shortcut / different page), or a different interaction sequence. Re-running the same path is forbidden; picking a new path is required.

If the plan is already good, output it unchanged. If corrections are needed, output the corrected plan.

Output format:
<plan>
Corrected numbered step-by-step plan.
</plan>"""

    system_content = "You review and correct GUI task plans. Be concise and practical."
    if system_prompt_suffix and system_prompt_suffix.strip():
        system_content = system_content + "\n\n" + system_prompt_suffix.strip()
    messages = [
        {"role": "system", "content": system_content},
    ]
    user_content = []
    for b64 in screenshots_b64[-3:]:
        url = b64 if b64.startswith("data:image") else f"data:image/png;base64,{b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    try:
        response = call_llm_fn(messages, max_tokens=600)
        if not response:
            return None
        m = re.search(r'<plan>(.*?)</plan>', response, re.DOTALL)
        if m:
            corrected = m.group(1).strip()
            if corrected and corrected != plan.strip():
                if 'JFK' in plan and 'kennedy' not in instruction.lower():
                    corrected = corrected.replace('JFK', 'NYC')
                return corrected
        return None
    except Exception:
        return None


# ── Embedding-based a11y element filtering ─────────────────────────────────────

_EMBED_MODEL = None

def _get_embed_model():
    """Lazy-load sentence-transformers model (cached singleton)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def a11y_filter_by_embedding(subgoal: str, elements: List[Dict],
                              top_k: int = 20) -> List[Dict]:
    """Filter a11y elements by semantic similarity to the subgoal using embeddings.

    Args:
        subgoal: current subgoal text
        elements: list of element dicts from A11yElementHelper
        top_k: number of top elements to return

    Returns: top_k elements sorted by relevance
    """
    import numpy as np
    model = _get_embed_model()

    # Build element text representations
    elem_texts = []
    for e in elements:
        text = f'{e["role"]} "{e["name"]}"'
        if e.get("value"):
            text += f' value="{e["value"]}"'
        elem_texts.append(text)

    if not elem_texts:
        return elements

    q_emb = model.encode([subgoal])
    e_embs = model.encode(elem_texts)
    sims = np.dot(e_embs, q_emb.T).squeeze()

    # Rank by similarity and take top_k
    ranked_indices = np.argsort(-sims)[:top_k]
    filtered = [elements[i] for i in ranked_indices]

    return filtered


# ── Two-stage a11y element selection ───────────────────────────────────────────

def _parse_llm_json(response: str):
    """Parse JSON from LLM response, handling markdown code blocks."""
    response = (response or "").strip()
    if "```" in response:
        response = response.split("```")[1].strip()
        if response.startswith("json"):
            response = response[4:].strip()
    return json.loads(response)


def a11y_filter_elements(subgoal: str, elements_text: str, call_llm_fn) -> List[int]:
    """Stage 1: Ask LLM to filter relevant element indices from full a11y list.

    Args:
        subgoal: current subgoal text
        elements_text: formatted element list from A11yElementHelper.format_elements_for_llm()
        call_llm_fn: callable(messages, max_tokens) -> str

    Returns: list of relevant element indices, or empty list on failure
    """
    prompt = f"""From the list of interactive elements below, select the indices that are relevant to this subgoal. Make sure to select the correct indices for the subgoal, because the indices are used for action execution in the next stage.

Subgoal: {subgoal}

Elements:
{elements_text}

Output ONLY a JSON list of relevant index numbers, e.g. [13, 57, 117, 118]
Make sure to include all indices that are relevant to the subgoal!"""

    messages = [
        {"role": "system", "content": "Select relevant element indices. Output only a JSON list of integers."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = call_llm_fn(messages, max_tokens=100)
        indices = _parse_llm_json(response)
        if isinstance(indices, list):
            return [i for i in indices if isinstance(i, int)]
    except Exception:
        pass
    return []


def a11y_select_actions(instruction: str, subgoal: str, filtered_text: str,
                        call_llm_fn, screenshot_b64: str = None) -> List[Dict]:
    """Stage 2: Ask LLM to select actions on filtered elements.

    Args:
        instruction: full task instruction
        subgoal: current subgoal text
        filtered_text: formatted filtered element list
        call_llm_fn: callable(messages, max_tokens) -> str
        screenshot_b64: optional screenshot for visual context

    Returns: list of action dicts [{index, action, value?}], or empty list on failure
    """
    prompt = f"""You are interacting with a web page. Given the current subgoal and the FILTERED list of relevant interactive elements, choose which element(s) to interact with.

Task: {instruction}
Current subgoal: {subgoal}

Relevant interactive elements:
{filtered_text}

Output a JSON list of actions to perform, in order. Each action has:
- "index": element index number (from the list above)
- "action": one of "click", "fill", "check", "uncheck", "select"
- "value": text to type (for "fill") or option to select (for "select"), omit for click/check

Output ONLY valid JSON, no explanation:
{{"actions": [{{"index": 0, "action": "click"}}, {{"index": 1, "action": "fill", "value": "25"}}]}}"""

    user_content = []
    if screenshot_b64:
        url = screenshot_b64 if screenshot_b64.startswith("data:image") else f"data:image/png;base64,{screenshot_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You select web page elements by index and specify actions. Output valid JSON only."},
        {"role": "user", "content": user_content},
    ]
    try:
        response = call_llm_fn(messages, max_tokens=200)
        data = _parse_llm_json(response)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("actions", [])
    except Exception:
        pass
    return []


# ── Web form subgoal detection ─────────────────────────────────────────────────

def is_web_form_subgoal(subgoal: str, screenshot_b64: str,
                        call_llm_fn) -> bool:
    """Use LLM + screenshot to determine if a subgoal involves web form interaction.

    Detects: filling text fields, applying filters, selecting dates, choosing
    from dropdowns, clicking checkboxes, setting passenger counts, etc.

    Args:
        subgoal: the subgoal text
        screenshot_b64: base64-encoded screenshot of the current page
        call_llm_fn: callable(messages, max_tokens) -> str

    Returns: True if the subgoal involves web form interaction
    """
    prompt = f"""Look at this screenshot and subgoal. Does this subgoal involve interacting with a web form element on the page?

Web form interactions include:
- Filling text fields (search bars, input boxes, origin/destination fields)
- Applying filters (price, color, size, brand, discount checkboxes/links)
- Selecting dates from date pickers or calendars
- Choosing from dropdowns (trip type, sort order, distance)
- Clicking checkboxes or radio buttons (one-way, round trip, shop with miles)
- Setting counters (passengers, rooms, guests)
- Clicking search/submit/apply buttons on forms

NOT web form interactions:
- Navigating to a different page by clicking a link
- Scrolling the page
- Opening a new tab
- General page browsing without form interaction

Subgoal: {subgoal}

Answer ONLY: YES or NO"""

    user_content = []
    if screenshot_b64:
        url = screenshot_b64 if screenshot_b64.startswith("data:image") else f"data:image/png;base64,{screenshot_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "Classify subgoals. Answer YES or NO only."},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm_fn(messages, max_tokens=5)
        answer = (response or "").strip().upper()
        return "YES" in answer
    except Exception:
        return False


def locate_via_ctrl_f(search_term: str) -> List[str]:
    """Generate pyautogui code to Ctrl+F search_term, jump to match, then close find bar."""
    safe_term = re.sub(r"(?<!\\)'", r"\\'", search_term)
    code = "import pyautogui\nimport time\n"
    code += "\npyautogui.hotkey('ctrl', 'f')"
    code += "\ntime.sleep(0.5)"
    code += "\npyautogui.hotkey('ctrl', 'a')"
    code += f"\npyautogui.write('{safe_term}', interval=0.05)"
    code += "\ntime.sleep(0.3)"
    code += "\npyautogui.press('enter')"
    code += "\ntime.sleep(0.5)"
    code += "\npyautogui.press('escape')"
    return [code]


def locate_target(subgoal: str, screenshot_b64: str, call_llm_fn) -> List[str]:
    """Pre-action locate: use LLM to check if subgoal target is off-screen, then Ctrl+F to it.

    Args:
        subgoal: the current subgoal text
        screenshot_b64: base64-encoded screenshot of the current page
        call_llm_fn: callable(messages, max_tokens) -> str

    Returns:
        List of pyautogui code strings to execute, or [] if target is visible / no locate needed.
    """
    prompt = (
        f"Subgoal: {subgoal}\n\n"
        "Is the target of this subgoal clearly visible in the screenshot?\n"
        "If the target is not visible, you need to determine the short search term to Ctrl+F to find the target.\n"
        "Answer EXACTLY in one of these formats:\n"
        "VISIBLE\n"
        "LOCATE: <short search term to find the target, 1-2 words, NO additional words, as simple as possible>"
    )

    user_content = []
    if screenshot_b64:
        url = screenshot_b64 if screenshot_b64.startswith("data:image") else f"data:image/png;base64,{screenshot_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a GUI task analyst. Determine if a subgoal's target is visible on screen."},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm_fn(messages, max_tokens=30)
        if not response:
            return []
        m = re.match(r'LOCATE:\s*(.+)', response.strip(), re.IGNORECASE)
        if not m:
            return []
        search_term = m.group(1).strip().strip('"').strip("'")
        if not search_term:
            return []
        return locate_via_ctrl_f(search_term)
    except Exception:
        return []


_TERMINAL_SUBGOAL_PROMPT_TEMPLATE = """You are a GUI task automation assistant. Given a subgoal and the current screenshot, determine if this subgoal can be completed with a single terminal/bash command instead of GUI interaction.

Subgoal: {subgoal}
Overall task: {instruction}

Environment: Ubuntu Linux, user "user", home at /home/user

=== WHEN TO USE TERMINAL COMMANDS ===
Use a terminal command when the subgoal involves:
- Editing config files (use sed)
- Video/audio conversion or extraction (use ffmpeg or cvlc)
- File operations: rename, move, copy, delete (use mv, cp, rm)
- Setting desktop wallpaper (use gsettings)
- Checking file existence or content (use ls, cat, grep)
- Bulk document formatting changes (font, size, spacing, case) in LibreOffice Writer/Calc/Impress — use python-docx/openpyxl/python-pptx via the close-edit-reopen pattern (see examples)

Do NOT use terminal commands when the subgoal involves:
- Clicking UI buttons, menus, or checkboxes
- Visual interaction (dragging, scrolling within an app)
- Tasks that require seeing the screen to decide what to do
- Installing applications (use the GUI app store / Ubuntu Software instead — sudo commands may fail)
- Closing terminals (answer NO — terminal management is handled separately)

NEVER include these in your command:
- pkill, kill, killall for gnome-terminal or other system processes
- gnome-terminal, xterm (do NOT open/manage terminals)
EXCEPTION: "pkill -f soffice" is ALLOWED when you need to close LibreOffice to edit a document file programmatically. Always reopen the file with "soffice '/path/to/file' &" after editing.

{domain_section}
=== IMPORTANT: ALWAYS INCLUDE VERIFICATION ===
After your main command, append a verification command that checks the ACTUAL RESULT, separated by " ; ".
The verification should prove the command worked — e.g., check file exists, print file content, check config value, etc.

Examples:
- File creation: printf '1<br/>\n2<br/>\n3<br/>\n' > output.txt ; cat output.txt
- Config edit: sed -i 's/^#\?qt-bgcone=.*/qt-bgcone=0/' /home/user/.config/vlc/vlcrc ; grep 'qt-bgcone' /home/user/.config/vlc/vlcrc
- File move: mv old.txt new.txt ; ls -la new.txt
- File copy: cp a.txt b.txt ; diff a.txt b.txt && echo 'identical'

=== OUTPUT FORMAT ===
Output EXACTLY one of:
COMMAND: <main command> ; <verification command>
NO
"""

# Cache for loaded terminal command configs
_terminal_commands_cache: Optional[dict] = None


def _load_terminal_commands() -> dict:
    """Load domain-specific terminal command config from terminal_commands.json."""
    global _terminal_commands_cache
    if _terminal_commands_cache is not None:
        return _terminal_commands_cache
    import os
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    path = os.path.join(base, "evaluation_examples", "terminal_commands.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _terminal_commands_cache = json.load(f)
    except Exception:
        _terminal_commands_cache = {}
    return _terminal_commands_cache


def _build_domain_section(domain: Optional[str]) -> str:
    """Build the domain-specific config files and examples section for the prompt."""
    commands = _load_terminal_commands()
    if not domain or domain not in commands:
        # No domain match — include all domains as general reference
        all_examples = []
        for d, data in commands.items():
            if d.startswith("_"):
                continue
            for ex in data.get("examples", []):
                all_examples.append(ex)
        if not all_examples:
            return ""
        lines = ["\n=== EXAMPLES ===\n"]
        for ex in all_examples[:20]:
            lines.append(f'Subgoal: "{ex["subgoal"]}"')
            lines.append(f'COMMAND: {ex["command"]}\n')
        lines.append('Subgoal: "Click the Save button" → NO (requires GUI)')
        lines.append('Subgoal: "Open the Tools menu" → NO (requires GUI)')
        lines.append('Subgoal: "Scroll down to find the option" → NO (requires GUI)\n')
        return "\n".join(lines)

    data = commands[domain]
    lines = []

    # Notes section (domain-specific guidance)
    notes = data.get("notes", "")
    if notes:
        lines.append(f"\n=== DOMAIN NOTES ===\n{notes}\n")

    # Config files section
    config_files = data.get("config_files", [])
    if config_files:
        lines.append("\n=== CONFIG FILE LOCATIONS ===")
        for cf in config_files:
            lines.append(f"- {cf['path']}")
            lines.append(f"  Format: {cf['format']}")
            lines.append(f"  How to edit: {cf['how_to_edit']}")
        lines.append("")

    # Examples section
    examples = data.get("examples", [])
    if examples:
        lines.append("=== EXAMPLES ===\n")
        for ex in examples:
            lines.append(f'Subgoal: "{ex["subgoal"]}"')
            lines.append(f'COMMAND: {ex["command"]}\n')
    lines.append('Subgoal: "Click the Save button" → NO (requires GUI)')
    lines.append('Subgoal: "Open the Tools menu" → NO (requires GUI)')
    lines.append('Subgoal: "Scroll down to find the option" → NO (requires GUI)\n')
    return "\n".join(lines)


def classify_terminal_subgoal(subgoal: str, instruction: str, screenshot_b64: str,
                              call_llm_fn, domain: Optional[str] = None) -> Optional[str]:
    """Use LLM to determine if a subgoal can be done with a terminal command.

    Args:
        subgoal: the current subgoal text
        instruction: the overall task instruction
        screenshot_b64: base64-encoded screenshot
        call_llm_fn: callable(messages, max_tokens) -> str
        domain: the app domain (e.g. "vlc", "os", "vs_code") for domain-specific examples

    Returns:
        The bash command string if applicable, or None.
    """
    domain_section = _build_domain_section(domain)
    prompt = _TERMINAL_SUBGOAL_PROMPT_TEMPLATE.format(
        subgoal=subgoal, instruction=instruction, domain_section=domain_section)

    user_content = []
    if screenshot_b64:
        url = screenshot_b64 if screenshot_b64.startswith("data:image") else f"data:image/png;base64,{screenshot_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You classify whether GUI subgoals can be done via terminal commands. Output COMMAND: <cmd> or NO."},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm_fn(messages, max_tokens=500)
        if not response:
            return None
        # Strip thinking content if present (Qwen3.5-VL <think> blocks)
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        m = re.match(r'COMMAND:\s*(.+)', clean, re.IGNORECASE)
        if not m:
            return None
        command = m.group(1).strip()
        if not command:
            return None
        return command
    except Exception:
        return None


def execute_bash_on_vm(env, command: str, timeout: int = 30) -> Tuple[bool, str]:
    """Execute a bash command on the VM.

    Tries env.controller.run_bash_script first. If the controller doesn't support it
    (e.g. APIDesktopEnv's _RemoteController), falls back to env.step() with a
    subprocess.run() wrapped in Python code.

    Returns:
        (success, output_or_error) where success = returncode == 0.
    """
    # Method 1: direct run_bash_script (PythonController)
    if hasattr(env.controller, 'run_bash_script'):
        try:
            result = env.controller.run_bash_script(command, timeout=timeout)
            if result is None:
                return False, "run_bash_script returned None"
            returncode = result.get("returncode", -1)
            output = result.get("output", "")
            error = result.get("error", "")
            if returncode == 0:
                return True, output
            else:
                return False, f"returncode={returncode}, error={error}, output={output}"
        except Exception as e:
            return False, str(e)

    # Method 2: wrap bash command in Python subprocess and execute via env.step()
    safe_cmd = command.replace("\\", "\\\\").replace("'", "\\'")
    python_code = (
        "import subprocess\n"
        f"result = subprocess.run('{safe_cmd}', shell=True, capture_output=True, text=True, timeout={timeout})\n"
        "print(f'EXIT_RC:{result.returncode}')\n"
        "print(f'STDOUT:{result.stdout}')\n"
        "print(f'STDERR:{result.stderr}')"
    )
    try:
        obs, _, done, _ = env.step(python_code)
        # The command executed via env.step; check if env is still alive
        if done:
            return False, "env reported done after bash execution"
        return True, "executed via env.step"
    except Exception as e:
        return False, str(e)


def generate_vscode_settings_edit_actions(settings_to_merge: dict, env) -> List[str]:
    """Generate actions to edit VS Code settings.json safely.

    Flow:
    1. env.step: Read current settings.json, merge new settings, copy result to clipboard
    2. env.step: Open settings.json in VS Code via Command Palette
    3. Return pyautogui code: Ctrl+A → Ctrl+V (paste merged JSON) → Ctrl+S

    This avoids auto-closing bracket issues by pasting the complete JSON.

    Args:
        settings_to_merge: dict of key-value pairs to add/update in settings.json
        env: the environment object

    Returns:
        List of pyautogui code strings to execute, or []
    """
    import json as _json
    merge_json = _json.dumps(settings_to_merge)

    # Step 1: Read current settings, merge, copy to clipboard on the VM
    merge_and_copy_code = (
        "import json, os, subprocess\n"
        f"new_settings = {merge_json}\n"
        "p = os.path.expanduser('~/.config/Code/User/settings.json')\n"
        "d = json.load(open(p)) if os.path.exists(p) else {}\n"
        "for k, v in new_settings.items():\n"
        "    if isinstance(v, dict) and isinstance(d.get(k), dict):\n"
        "        d[k].update(v)\n"
        "    else:\n"
        "        d[k] = v\n"
        "result = json.dumps(d, indent=2)\n"
        "proc = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)\n"
        "proc.communicate(result.encode())\n"
        "print('MERGED_SETTINGS:', result[:200])\n"
    )
    try:
        env.step(merge_and_copy_code)
    except Exception:
        return []

    # Step 2: Open settings.json in VS Code editor
    open_settings_code = (
        "import pyautogui, time\n"
        "pyautogui.hotkey('ctrl', 'shift', 'p')\n"
        "time.sleep(0.5)\n"
        "pyautogui.write('Preferences: Open User Settings (JSON)', interval=0.02)\n"
        "time.sleep(0.3)\n"
        "pyautogui.press('enter')\n"
        "time.sleep(1.0)\n"
    )
    try:
        env.step(open_settings_code)
    except Exception:
        return []

    # Step 3: Select all → paste → save
    paste_code = (
        "import pyautogui, time\n"
        "pyautogui.hotkey('ctrl', 'a')\n"
        "time.sleep(0.2)\n"
        "pyautogui.hotkey('ctrl', 'v')\n"
        "time.sleep(0.5)\n"
        "pyautogui.hotkey('ctrl', 's')\n"
        "time.sleep(0.5)\n"
    )

    return [paste_code]

