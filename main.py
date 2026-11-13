
import os
from dotenv import load_dotenv
from ddgs import DDGS
from pydantic import BaseModel
from agents import (
    Agent, Runner, function_tool, set_tracing_disabled,
    ModelSettings, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    input_guardrail, output_guardrail
)
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()
set_tracing_disabled(True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env file")
model = LitellmModel(
    model="gemini/gemini-flash-lite-latest",
    api_key=api_key
)
# STRUCTURED OUTPUT
class ResearchReport(BaseModel):
    topic: str
    points: list[str]
# WEB SEARCH TOOL
@function_tool
def web_search(query: str) -> str:
    """Search the internet for research information."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        return "\n".join(
            f"Title: {r['title']}\nLink: {r['href']}\nInfo: {r['body']}"
            for r in results
        )
    except Exception:
        return "Web search is temporarily unavailable."
# INPUT GUARDRAIL
@input_guardrail(run_in_parallel=False)
def check_input(ctx, agent, user_input):
    blocked = ["password", "api key", "secret"]
    is_blocked = any(
        word in str(user_input).lower() for word in blocked
    )
    return GuardrailFunctionOutput(
        output_info="Input security check",
        tripwire_triggered=is_blocked
    )
# OUTPUT GUARDRAIL
@output_guardrail
def check_output(ctx, agent, output):
    blocked = ["password", "api key", "secret"]

    is_blocked = any(
        word in str(output).lower() for word in blocked
    )
    return GuardrailFunctionOutput(
        output_info="Output security check",
        tripwire_triggered=is_blocked
    )
# AGENT INSTRUCTIONS
instructions = """
You are a professional research assistant.
Rules:
1. Search the internet for the user's topic.
2. Use reliable information.
3. Write in simple Roman English.
4. Return only the topic name and 3 important points.
5. Each point must be short and clear.
6. Do not include summary or extra explanations.
7. Follow the required JSON structure.
"""
# AI AGENT
agent = Agent(
    name="Research Assistant",
    instructions=instructions,
    model=model,
    tools=[web_search],
    input_guardrails=[check_input],
    output_guardrails=[check_output],
    output_type=ResearchReport,
    model_settings=ModelSettings(
        max_tokens=300,
        include_usage=True
    )
)
# CONTINUOUS QUESTIONS
if __name__ == "__main__":
    print("\n--- RESEARCH ASSISTANT ---")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Enter topic: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a topic.\n")
            continue

        try:
            result = Runner.run_sync(agent, question)
            report = result.final_output
            print("\nTopic:", report.topic)
            for i, point in enumerate(report.points[:3], 1):
                print(f"{i}. {point}")
            usage = result.context_wrapper.usage
            print("\nToken Usage:")
            print("Input:", usage.input_tokens)
            print("Output:", usage.output_tokens)
            print("Total:", usage.total_tokens)

        except InputGuardrailTripwireTriggered:
            print("\nBlocked: Your input is not allowed.")
        except OutputGuardrailTripwireTriggered:
            print("\nBlocked: Agent output did not pass security checks.")
        except Exception as e:
            print("\nError:", str(e))
        print()