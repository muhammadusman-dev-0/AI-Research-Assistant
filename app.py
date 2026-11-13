from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agents import Runner, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from main import agent

app = FastAPI()

class TopicRequest(BaseModel):
    topic: str

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/research")
async def research(req: TopicRequest):
    try:
        result = await Runner.run(agent, req.topic)
        report = result.final_output
        usage = result.context_wrapper.usage

        return {
            "blocked": False,
            "topic": report.topic,
            "points": report.points,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
        }

    except InputGuardrailTripwireTriggered:
        return {"blocked": True, "message": "Your input is not allowed."}
    except OutputGuardrailTripwireTriggered:
        return {"blocked": True, "message": "Agent output did not pass security checks."}
    except Exception as e:
        return {"blocked": True, "message": f"Error: {str(e)}"}
