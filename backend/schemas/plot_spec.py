from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class PlotSpec(BaseModel):
    """
    Structured visualization specification from LLM intent parsing.
    Bridge between natural language requests and plotting functions.
    """
    chart_type: str = Field(..., description="bar, line, scatter, histogram, pie, box, heatmap")
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    title: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    confidence: str = Field(default="high", description="high, medium, low")
    reasoning: Optional[str] = None     # Why this chart was chosen


class VisualizationGuidance(BaseModel):
    """
    Educational response when user asks for guidance instead of chart generation.
    """
    explanation: str
    recommended_chart: str
    reasoning: str
    suggested_x_axis: Optional[str] = None
    suggested_y_axis: Optional[str] = None
    alternatives: Optional[list[str]] = None


class IntentParserResult(BaseModel):
    """
    Result from LLM intent parsing.
    """
    mode: str  # "visualization" | "guidance" | "unclear"
    plot_spec: Optional[PlotSpec] = None
    guidance: Optional[VisualizationGuidance] = None
    error: Optional[str] = None