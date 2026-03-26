"""
Artist Intent Parser - Converts natural language visualization requests to structured PlotSpec objects.

This module bridges natural language requests with the visualization pipeline:
1. Detects whether user wants a visualization or educational guidance
2. Parses the request using an LLM-based intent parser
3. Converts responses to Pydantic models (IntentParserResult, PlotSpec, VisualizationGuidance)
4. Validates output before passing to plotting functions
"""

import json
import logging
from typing import Optional
from pathlib import Path

import httpx
from pydantic import ValidationError

from backend.schemas.plot_spec import (
    IntentParserResult,
    PlotSpec,
    VisualizationGuidance,
)

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"


def load_parser_prompt() -> str:
    """Load the artist parser prompt template from prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "artist_parser_prompt.txt"
    
    if not prompt_path.exists():
        logger.warning(f"Parser prompt not found at {prompt_path}, using inline template")
        return _get_default_prompt()
    
    with open(prompt_path, "r") as f:
        return f.read()


def _get_default_prompt() -> str:
    """Fallback inline prompt template if file not found."""
    return """You are a Data Visualization Intent Parser. Determine if the user wants:
1. VISUALIZATION MODE: An actual chart
2. GUIDANCE MODE: Educational advice on visualization

Available columns: {columns}

If user wants a chart, return:
{{"mode": "visualization", "plot_spec": {{"chart_type": "...", "x_axis": "...", "y_axis": "...", "confidence": "..."}}}}

If asking "how to visualize", return guidance mode.
If unclear, return "unclear" mode.

USER REQUEST: {user_prompt}

Return ONLY valid JSON."""


async def parse_visualization_intent(
    user_prompt: str,
    columns: list[str],
    model: str = DEFAULT_MODEL,
    ollama_host: str = OLLAMA_HOST,
) -> IntentParserResult:
    """
    Parse natural language visualization request using LLM intent parser.
    
    Args:
        user_prompt: User's natural language request
        columns: Available dataset columns
        model: Ollama model to use
        ollama_host: Ollama endpoint URL
        
    Returns:
        IntentParserResult with mode, plot_spec or guidance, and optional error
    """
    try:
        # Load and format prompt
        prompt_template = load_parser_prompt()
        columns_str = ", ".join(columns)
        
        system_prompt = prompt_template.format(
            columns=columns_str,
            user_prompt=user_prompt
        )
        
        logger.debug(f"Calling {model} for intent parsing")
        
        # Call Ollama
        response = await _call_ollama(system_prompt, model, ollama_host)
        
        # Extract JSON from response
        result_dict = _extract_json_from_response(response)
        
        # Validate and convert to appropriate model
        return _validate_and_convert_response(result_dict, columns)
        
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}", exc_info=True)
        return IntentParserResult(
            mode="unclear",
            error=f"Failed to parse request: {str(e)}"
        )


async def _call_ollama(
    prompt: str,
    model: str,
    ollama_host: str
) -> str:
    """Call Ollama API with the prepared prompt."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ollama_host}/api/generate",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")


def _extract_json_from_response(response: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    # Try direct JSON parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in response (LLM might include explanations)
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not extract valid JSON from response: {response}")


def _validate_and_convert_response(
    result_dict: dict,
    available_columns: list[str]
) -> IntentParserResult:
    """
    Validate LLM response and convert to appropriate Pydantic model.
    
    Args:
        result_dict: Parsed JSON from LLM
        available_columns: List of valid column names for validation
        
    Returns:
        IntentParserResult with appropriate mode
        
    Raises:
        ValidationError if response structure is invalid
    """
    mode = result_dict.get("mode", "unclear")
    
    if mode == "visualization":
        # Validate and convert PlotSpec
        plot_spec_dict = result_dict.get("plot_spec", {})
        plot_spec_dict = _validate_columns_in_spec(plot_spec_dict, available_columns)
        
        try:
            plot_spec = PlotSpec(**plot_spec_dict)
            return IntentParserResult(
                mode="visualization",
                plot_spec=plot_spec
            )
        except ValidationError as e:
            logger.warning(f"PlotSpec validation failed: {e}")
            return IntentParserResult(
                mode="unclear",
                error=f"Invalid visualization specification: {str(e)}"
            )
    
    elif mode == "guidance":
        # Validate and convert VisualizationGuidance
        guidance_dict = result_dict.get("guidance", {})
        
        try:
            guidance = VisualizationGuidance(**guidance_dict)
            return IntentParserResult(
                mode="guidance",
                guidance=guidance
            )
        except ValidationError as e:
            logger.warning(f"Guidance validation failed: {e}")
            return IntentParserResult(
                mode="unclear",
                error=f"Invalid guidance specification: {str(e)}"
            )
    
    else:
        # Unclear or unknown mode
        error = result_dict.get("error", "Cannot determine visualization intent from request.")
        return IntentParserResult(
            mode="unclear",
            error=error
        )


def _validate_columns_in_spec(spec_dict: dict, available_columns: list[str]) -> dict:
    """
    Validate that x_axis and y_axis reference actual columns.
    Downgrades confidence if columns are invalid.
    """
    spec = spec_dict.copy()
    issues = []
    
    # Check x_axis
    if "x_axis" in spec and spec["x_axis"]:
        if spec["x_axis"] not in available_columns:
            issues.append(f"x_axis '{spec['x_axis']}' not in dataset")
    
    # Check y_axis
    if "y_axis" in spec and spec["y_axis"]:
        if spec["y_axis"] not in available_columns:
            issues.append(f"y_axis '{spec['y_axis']}' not in dataset")
    
    # Downgrade confidence if issues found
    if issues:
        current_confidence = spec.get("confidence", "high")
        
        # Map confidence to lower levels
        confidence_map = {
            "high": "medium",
            "medium": "low",
            "low": "low"
        }
        spec["confidence"] = confidence_map.get(current_confidence, "low")
        
        reasoning = spec.get("reasoning", "")
        spec["reasoning"] = f"{reasoning} [Note: {', '.join(issues)}]"
        
        logger.warning(f"Column validation issues in plot spec: {', '.join(issues)}")
    
    return spec


def _should_request_guidance(prompt: str) -> bool:
    """
    Heuristic to detect if user is asking for guidance vs visualization.
    Used as fallback if LLM doesn't return proper mode.
    """
    guidance_keywords = {
        "how", "what", "should", "can", "best", "recommend",
        "guide", "advice", "help", "way to", "approach"
    }
    
    prompt_lower = prompt.lower()
    
    # Check for question marks or guidance keywords
    if "?" in prompt:
        for keyword in guidance_keywords:
            if keyword in prompt_lower:
                return True
    
    # Check for explicit guidance phrases
    guidance_phrases = [
        "how should i", "what chart", "how can i",
        "best way to", "what's the best", "help me visualize"
    ]
    
    for phrase in guidance_phrases:
        if phrase in prompt_lower:
            return True
    
    return False
