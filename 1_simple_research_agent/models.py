from typing import TypedDict, List, Any

class EventDetails(TypedDict, total=False):
    content: str
    tool_call: str
    tool_result: str
    thinking: str
    text: str
    is_final: bool
    response: str

# 2. Define the outer event data structure
class EventData(TypedDict):
    type: str                  # Explicitly allow the 'type' key
    timestamp: str             # Explicitly allow 'timestamp' (change to float if it is numeric)
    details: EventDetails

class ExecutionTrace(TypedDict):
    events: List[Any]
    tools_called: List[Any]
    reasoning_steps: List[str]
    timestamps: List[str]