"""
Build per-task JSONL files from osunlp/Multimodal-Mind2Web huggingface dataset.
Each output file is named: conversation_{domain}_{id}.jsonl

- Groups actions into tasks by (domain, confirmed_task)
- Sorts by target_action_index
- For each task, writes:
  1) Task header:
     {"conversation_id": "domain_id", "task_description": confirmed_task, "total_rounds": N_actions + 1}
  2) One line per action:
     {"round_number": k, "messages": [...], "response": {"content": "... markdown with Action json ..."}}
  3) Final stop round

Keys used from each example:
['operation', 'domain', 'confirmed_task', 'screenshot', 'target_action_index', 'target_action_reprs']

"""

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import load_dataset
from openai import OpenAI

# -----------------------------------------------------------------------------
# Config switches
# -----------------------------------------------------------------------------
# If True, the JSON in the markdown code block will contain NaN (non-standard).
# Otherwise we use strict JSON: element_id = null, reasoning = "".
USE_JSON_NAN = False

# Whether to try a tool LLM for page description. If you have your DirectVLLMModel,
# set this True and implement `create_direct_vllm_model` below.
USE_TOOL_LLM = True

# -----------------------------------------------------------------------------
# (Optional) Tool LLM plumbing (safe to leave as no-op)
# -----------------------------------------------------------------------------
class DirectVLLMModel:
    """Direct vLLM model wrapper that can be used without qwen_agent"""
    
    def __init__(self, model_name: str, server_url: str, api_key: str = "EMPTY", **kwargs):
        self.model_name = model_name
        self.server_url = server_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, functions: List[Dict] = None, function_call: str = "auto", **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
        }
        
        # # Add function calling if provided
        # if functions:
        #     call_params["functions"] = functions
        #     call_params["function_call"] = function_call
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response
        else:
            return response.choices[0].message


def create_direct_vllm_model(model_name: Optional[str] = None, max_tokens: int = 512) -> DirectVLLMModel:
    """Create a direct vLLM model instance (safe default)."""
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'ui-tars': 'http://localhost:8001/v1',
    }
    # Default to qwen2.5-vl
    key = model_name or 'qwen2.5-vl'
    resolved_name = model_name_map.get(key, key)
    server_url = model_server_map.get(key, 'http://localhost:8000/v1')
    return DirectVLLMModel(
        model_name=resolved_name,
        server_url=server_url,
        api_key="EMPTY",
        temperature=0.2,
        top_p=0.9,
        max_tokens=max_tokens,
    )

def load_tool_llm(max_tokens: int = 512) -> Optional[DirectVLLMModel]:
    if not USE_TOOL_LLM:
        return None
    try:
        return create_direct_vllm_model('qwen2.5-vl', max_tokens=max_tokens)
    except Exception as e:
        logging.warning(f"Tool LLM unavailable, falling back to stub: {e}")
        return None

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def to_base64_image(data: Any) -> Optional[str]:
    """
    Try to convert `data` (various possible formats) into base64-encoded PNG string (no header).
    Accepted inputs:
      - bytes (already PNG/JPEG)
      - str (path to image or already base64 starting with 'data:image')
      - dict from HF datasets image feature (may contain 'bytes' or 'path')
      - PIL.Image.Image
    Returns base64 string (no 'data:image/...;base64,' prefix), or None if fails.
    """
    try:
        # Already a data URL?
        if isinstance(data, str):
            if data.startswith("data:image/"):
                # Extract the base64 payload
                return data.split("base64,", 1)[-1]
            if os.path.isfile(data):
                with open(data, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            # Might already be raw base64 without header
            # Try decode-encode roundtrip to validate
            try:
                base64.b64decode(data, validate=True)
                return data
            except Exception:
                pass

        # Hugging Face Image dict-like
        if isinstance(data, dict):
            if "bytes" in data and data["bytes"] is not None:
                return base64.b64encode(data["bytes"]).decode("utf-8")
            if "path" in data and data["path"] and os.path.isfile(data["path"]):
                with open(data["path"], "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        # Raw bytes
        if isinstance(data, (bytes, bytearray)):
            return base64.b64encode(bytes(data)).decode("utf-8")

        # PIL Image?
        try:
            from PIL import Image
            if isinstance(data, Image.Image):
                buf = io.BytesIO()
                data.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

    except Exception as e:
        logging.warning(f"Failed to convert screenshot to base64: {e}")

    return None

def generate_page_description(tool_llm: Optional[DirectVLLMModel], image_base64: Optional[str]) -> str:
    """
    Generate a description for the current page. If tool_llm is None or fails, return a fallback string.
    """
    fallback = "Current page state - analyze this and decide what to do next."
    if not tool_llm or not image_base64:
        return fallback

    try:
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that analyzes web page screenshots and provides clear, concise descriptions of what you see.'
            },
            {
                'role': 'user',
                'content': [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": "Please describe this web page screenshot. Include the main content, visible controls, and the overall purpose of the page."
                    }
                ]
            }
        ]
        resp = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", None)
        if not text:
            return fallback
        # Sanitize odd wrappers
        text = text.replace("\"text\": \"{'role': 'assistant', 'content': '", "").replace("'}\"", "")
        return text[:2000] if text else fallback
    except Exception as e:
        logging.warning(f"Tool LLM description failed: {e}")
        return fallback

SYSTEM_PROMPT = """You are a GUI automation agent that can interact with web pages and applications using the ReAct (Reasoning and Acting) paradigm.

IMPORTANT: You MUST output your actions in structured JSON format that can be parsed directly. Use the function calling mechanism to execute actions.

Your task is to:
1. Analyze the current state of the page (including numerical labels on web elements)
2. Think through what needs to be done (Reasoning)
3. Determine the appropriate action to take (Acting)
4. Output the action in structured JSON format that can be parsed directly.

WORKFLOW GUIDELINES:
- If your previous action is type, then you must click related pages or scroll pages to find the information you need.
- When you need to search for information: Directly type your search query, then click the search button.
- After clicking an element, if you need to interact with it further (like typing), do so immediately
- Don't repeat the same action multiple times - if something doesn't work, try a different approach
- ALWAYS use function calling to execute actions - do not describe actions in text
- If the current page has no results, you must adjust your search term, especially make your search term simpler and try again.
- Pay attention to images; for questions about shape, color, location, etc., you can answer according to the images.

ACTION GUIDELINES:
1) To input text, NO need to click textbox first; directly type content. After typing, the system automatically hits ENTER. Sometimes you should click the search button to apply filters.
2) Distinguish textbox vs. search button; don't type into a button! If no textbox is found, click a search control first to reveal it.
3) Execute only one action per iteration.
4) STRICTLY avoid repeating the same action if the webpage remains unchanged. Continuous use of Wait is NOT allowed.
5) Select "stop" only at the very end when the task is actually complete.
6) Finish all sub-goals (city/people/dates/etc. for bookings).
7) The task is finished only when the final target is visible and correct.

WEB BROWSING GUIDELINES:
1) Ignore Login/Sign-in/Donate unless essential.
2) YouTube allowed but do not play videos. Clicking to download PDF is allowed.
3) Use filter/sort + scroll to satisfy highest/cheapest/earliest etc.
4) Pay attention to images for visual questions.

Available actions:
- click: {element_id (NaN), description (string), reasoning (string)}
- type: {text (string), element_id (NaN), field_description (string), reasoning (string)}
- select: {description (string), text (string), reasoning (string)}
- stop: {answer (string), reasoning (string)}

CRITICAL REQUIREMENTS:
1. ALWAYS use function calling - never plain text.
2. Provide clear reasoning (can be brief).
3. Only one action at a time.
4. For click/type, set element_id to NaN (or null placeholder in data generation).
"""

def prepare_messages(image_b64: Optional[str], intent: str, tool_llm: Optional[DirectVLLMModel]) -> List[Dict[str, Any]]:
    """Build messages for the LLM with (image + description) + task text."""
    page_description = generate_page_description(tool_llm, image_b64)

    messages: List[Dict[str, Any]] = []
    messages.append({'role': 'system', 'content': SYSTEM_PROMPT})

    # current screenshot
    user_content: List[Dict[str, Any]] = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        })
    user_content.append({"type": "text", "text": page_description})
    messages.append({'role': 'user', 'content': user_content})

    # task intent & reminders
    messages.append({
        'role': 'user',
        'content': f"""**Current task:** {intent}

IMPORTANT REMINDERS:
- Specify the number label of the item you want to interact with in the description.
- Don't repeat the same action if the page is unchanged.
- If previous action is TYPE, consider CLICK/SCROLL next.
- If search yields no results, simplify and try again.
- Provide an answer within remaining steps.
- Only output one action at a time.

What would you like to do next?"""
    })
    return messages

# -----------------------------------------------------------------------------
# Parsing target_action_reprs
# -----------------------------------------------------------------------------
_repr_pat = re.compile(
    r"^\s*\[(?P<etype>[^\]]+)\]\s*(?P<desc>.*?)\s*->\s*(?P<op>CLICK|TYPE|SELECT|STOP)(?::\s*(?P<val>.*))?$",
    flags=re.IGNORECASE
)

def parse_action_repr(s: str) -> Dict[str, Optional[str]]:
    """
    Parse a string like:
      "[heading] CAR -> CLICK"
      "[combobox] Enter pick up city ... -> TYPE: Brooklyn Central"
      "[dropdown] Size -> SELECT: Medium"
      "... -> STOP"
    Return dict with keys: etype, desc, op (upper), val (optional).
    """
    m = _repr_pat.match(s.strip())
    if not m:
        # Fallback: try a looser split
        left, sep, right = s.partition("->")
        etype, desc = "element", left.strip()
        op = right.strip().upper() if right else "CLICK"
        val = None
        if ":" in op:
            op, _, v = op.partition(":")
            val = v.strip()
        return {"etype": etype, "desc": desc, "op": op.strip().upper(), "val": val}
    gd = m.groupdict()
    return {
        "etype": gd.get("etype"),
        "desc": gd.get("desc"),
        "op": (gd.get("op") or "").upper(),
        "val": gd.get("val")
    }

def op_to_name(op: str) -> str:
    m = (op or "").strip().lower()
    if m in {"click", "type", "select", "stop"}:
        return m
    return "click"

def build_arguments_from_repr(op_name: str, repr_texts: List[str], confirmed_task) -> Dict[str, Any]:
    """
    From a list of target_action_reprs, pick the one matching op_name if possible,
    then build the arguments dict required by your schema.
    """
    parsed = [parse_action_repr(s) for s in (repr_texts or [])]
    # Prefer the repr whose op matches
    chosen = None
    for r in parsed:
        if op_to_name(r.get("op", "")) == op_name:
            chosen = r
            break
    if chosen is None:
        chosen = parsed[0] if parsed else {"etype": "element", "desc": "", "op": op_name.upper(), "val": None}

    etype = chosen.get("etype") or "element"
    desc = chosen.get("desc") or ""
    val = chosen.get("val") or ""

    # Element-id & reasoning placeholders
    element_id_val = float('nan') if USE_JSON_NAN else None
    reasoning_val = ""  # keep empty as requested

    if op_name == "click":
        return {
            "element_id": element_id_val,
            "description": f"[{etype}] {desc}".strip(),
            "reasoning": reasoning_val
        }
    elif op_name == "type":
        return {
            "text": val,  # e.g. "Bryce Canyon National Park"
            "element_id": element_id_val,
            "field_description": f"[{etype}] {desc}".strip(),  # e.g. "[textbox] Search location ..."
            "reasoning": reasoning_val
        }
    elif op_name == "select":
        return {
            "description": f"[{etype}] {desc}".strip(),  # e.g. "[combobox] Model"
            "text": val,  # e.g. "e-tron"
            "reasoning": reasoning_val
        }
    elif op_name == "stop":
        return {
            "answer": f"Task completed successfully: {confirmed_task}",
            "reasoning": reasoning_val
        }
    else:
        # default to click
        return {
            "element_id": element_id_val,
            "description": f"[{etype}] {desc}".strip(),
            "reasoning": reasoning_val
        }

def wrap_action_markdown(name: str, arguments: Dict[str, Any]) -> str:
    """
    Build the markdown string for response.content with the **Action:** block and ```json fenced code.
    If USE_JSON_NAN=False, we emit strict JSON (null, true/false, numbers, strings).
    If USE_JSON_NAN=True, allow NaN inside json.dumps (Python allows it), still many parsers accept it.
    """
    if USE_JSON_NAN:
        json_text = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    else:
        # Replace NaN with null for strict JSON
        clean_args = {}
        for k, v in arguments.items():
            if isinstance(v, float) and str(v) == "nan":
                clean_args[k] = None
            else:
                clean_args[k] = v
        json_text = json.dumps({"name": name, "arguments": clean_args}, ensure_ascii=False)

    return f'\n**Action:**\n```json\n{json_text}\n```'

def infer_op_from_reprs(repr_texts: List[str]) -> str:
    """infer operation name from target_action_reprs; default to 'click'"""
    for s in (repr_texts or []):
        m = _repr_pat.match(s.strip())
        if m:
            op = (m.group("op") or "").strip().lower()
            if op in {"click", "type", "select", "stop"}:
                return op

    j = " ".join(repr_texts or []).lower()
    for key in ("type", "select", "stop", "click"):
        if f"-> {key}" in j or f": {key}" in j:
            return key
    return "click"

# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
@dataclass
class ExampleView:
    domain: str
    confirmed_task: str
    target_action_index: int
    target_action_reprs: List[str]
    screenshot_b64: Optional[str]

def example_to_view(ex: Dict[str, Any]) -> ExampleView:
    domain = ex.get("domain") or "unknown"
    confirmed_task = ex.get("confirmed_task") or ""
    t_idx = ex.get("target_action_index", 0) or 0
    reprs = ex.get("target_action_reprs") or []
    if isinstance(reprs, str):
        reprs = [reprs]
    b64 = to_base64_image(ex.get("screenshot"))
    return ExampleView(
        domain=str(domain),
        confirmed_task=str(confirmed_task),
        target_action_index=int(t_idx),
        target_action_reprs=list(reprs),
        screenshot_b64=b64
    )

def write_task_jsonl(
    out_dir: str,
    domain: str,
    conv_id: int,
    confirmed_task: str,
    ordered_examples: List[ExampleView],
    tool_llm: Optional[DirectVLLMModel]
):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"conversation_{domain}_{conv_id}.jsonl")

    total_rounds = len(ordered_examples) + 1  # + final stop
    header = {
        "conversation_id": f"{domain}_{conv_id}",
        "task_description": confirmed_task,
        "total_rounds": total_rounds
    }

    with open(fname, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        round_no = 1
        last_image_b64: Optional[str] = None

        for ev in ordered_examples:
            last_image_b64 = ev.screenshot_b64 or last_image_b64
            messages = prepare_messages(last_image_b64, confirmed_task, tool_llm)

            # name from target_action_reprs
            name = infer_op_from_reprs(ev.target_action_reprs)
            # arguments from target_action_reprs
            arguments = build_arguments_from_repr(name, ev.target_action_reprs, confirmed_task)
            content = wrap_action_markdown(name, arguments)

            line = {
                "round_number": round_no,
                "messages": messages,
                "response": {"content": content}
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            round_no += 1

        # final stop
        messages = prepare_messages(last_image_b64, confirmed_task, tool_llm)
        stop_args = build_arguments_from_repr("stop", ["[task] completion -> STOP"], confirmed_task)
        stop_content = wrap_action_markdown("stop", stop_args)
        stop_line = {
            "round_number": round_no,
            "messages": messages,
            "response": {"content": stop_content}
        }
        f.write(json.dumps(stop_line, ensure_ascii=False) + "\n")

    logging.info(f"Wrote {fname} (rounds={total_rounds})")

def process_streaming_dataset(out_dir: str, split: str = "train"):
    """
    Single pass over streaming dataset:
      - Group by (domain, confirmed_task)
      - Start a new file if target_action_index resets to 0 for an existing group
      - Flush all remaining groups at the end
    """
    logging.info("Loading dataset (streaming=True)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", streaming=True)
    stream = ds[split]

    # tool LLM (optional)
    tool_llm = load_tool_llm(max_tokens=768)

    # Track (domain -> running id counter)
    domain_to_next_id: Dict[str, int] = defaultdict(lambda: 1)

    # Active groups: key = (domain, confirmed_task) -> list of ExampleView
    active: Dict[Tuple[str, str], List[ExampleView]] = {}

    def flush_group(key: Tuple[str, str]):
        """Write out one group and advance domain id counter."""
        domain, confirmed_task = key
        exs = active.pop(key, [])
        if not exs:
            return
        # sort by target_action_index
        exs_sorted = sorted(exs, key=lambda e: e.target_action_index)
        conv_id = domain_to_next_id[domain]
        write_task_jsonl(out_dir, domain, conv_id, confirmed_task, exs_sorted, tool_llm)
        domain_to_next_id[domain] += 1

    for i, ex in enumerate(stream):
        ev = example_to_view(ex)
        key = (ev.domain, ev.confirmed_task)

        # If we see index == 0 for an existing active group, flush the old one first (start new task with same text)
        if ev.target_action_index == 0 and key in active and active[key]:
            flush_group(key)

        # Append to group
        active.setdefault(key, []).append(ev)

        # (Optional) you could also flush when detecting decreasing indices, etc.

        if (i + 1) % 500 == 0:
            logging.info(f"Scanned {i+1} examples... active groups={len(active)}")

    # End of stream: flush all
    for key in list(active.keys()):
        flush_group(key)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for JSONL files.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split.")
    parser.add_argument("--loglevel", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    process_streaming_dataset(out_dir=args.out_dir, split=args.split)

if __name__ == "__main__":
    main()
