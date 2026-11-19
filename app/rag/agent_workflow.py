
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from rag.retrieve import retrieve


class AgentState(TypedDict):
    question: str
    notebook: str
    status: str
    hits_count: int


class RequestContext(TypedDict):
    user_id: str


def validate_question(state: AgentState) -> dict:
    question = state["question"].strip()

    if not question:
        return {"status": "invalid_question"}

    return {
        "question": question,
        "status": "ready_for_retrieval"
    }


def route_question(state: AgentState) -> str:
    if state["status"] == "invalid_question":
        return "stop"

    return "continue"


def retrieve_authorized(
    state: AgentState,
    runtime: Runtime[RequestContext]
) -> dict:
    user_id = (runtime.context or {}).get("user_id")

    if not user_id:
        raise PermissionError("Authenticated user context required")

    hits = retrieve(
        user_id=user_id,
        notebook=state["notebook"],
        query=state["question"],
        top_k=4
    )

    return {
        "hits_count": len(hits),
        "status": "documents_found" if hits else "no_documents"
    }


builder = StateGraph(
    AgentState,
    context_schema=RequestContext
)

builder.add_node("validate_question", validate_question)
builder.add_node("retrieve_authorized", retrieve_authorized)

builder.add_edge(START, "validate_question")

builder.add_conditional_edges(
    "validate_question",
    route_question,
    {
        "stop": END,
        "continue": "retrieve_authorized"
    }
)

builder.add_edge("retrieve_authorized", END)

workflow = builder.compile()


if __name__ == "__main__":
    result = workflow.invoke({
        "question": "   ",
        "notebook": "canary-test-116",
        "status": "pending",
        "hits_count": 0
    })

    print(result)
