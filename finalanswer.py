@tool("final_answer")
def final_answer_tool(answer: str, source: str):
    """Returns a natural language response to the user in `answer`, and a
    `source` which provides citations for where this information came from.
    """
    return f"{answer}\n\nSource: {source}"


