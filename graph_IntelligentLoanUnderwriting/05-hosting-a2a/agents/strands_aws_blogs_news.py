import logging
import os
import asyncio
from strands import Agent, tool
from strands.multiagent.a2a import A2AServer
import uvicorn
from fastapi import FastAPI

from ddgs import DDGS
from ddgs.exceptions import RatelimitException, DDGSException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

runtime_url = os.environ.get('AGENTCORE_RUNTIME_URL', 'http://127.0.0.1:9000/')

@tool
async def fast_internet_search(keywords: str, max_results: int = 3) -> str:
    """Fast web search with timeouts.
    Args:
        keywords (str): Search query keywords
        max_results (int): Max results (default 3 for speed)
    Returns:
        Search results
    """
    try:
        # Add AWS-specific terms for better results
        aws_keywords = f"site:aws.amazon.com {keywords} AWS"
        
        # Use asyncio timeout for the search
        async def search_with_timeout():
            return DDGS().text(
                aws_keywords, 
                region="us-en", 
                max_results=max_results
            )
        
        results = await asyncio.wait_for(search_with_timeout(), timeout=8.0)
        
        if results:
            # Format results concisely
            formatted = []
            for i, result in enumerate(results[:max_results], 1):
                formatted.append(f"{i}. {result.get('title', 'No title')}\n   {result.get('href', '')}")
            
            return "\n".join(formatted)
        else:
            return "No AWS results found."
            
    except asyncio.TimeoutError:
        logger.warning(f"Search timeout for: {keywords}")
        return "Search timed out. Try a more specific query."
    except RatelimitException:
        logger.warning("Rate limit hit")
        return "Rate limit reached. Please try again in a moment."
    except (DDGSException, Exception) as e:
        logger.error(f"Search error: {e}")
        return f"Search unavailable: {str(e)[:50]}"

system_prompt = """
You are the Risk Analysis Manager responsible for evaluating risks and detecting potential fraud. Your duties include:

                Step 1: Risk Calculation Agent specialized in quantitative risk modeling:
                1. Calculate probability of default (PD)
                2. Estimate loss given default (LGD)
                3. Assess exposure at default (EAD)
                4. Compute risk-adjusted pricing
                5. Analyze portfolio concentration risks
                6. Generate risk scores and ratings

                Evaluate borrower risk profile. Risk Categories:
                - Credit risk (default probability)
                - Fraud risk (application authenticity)
                - Market risk (economic factors)
                - Operational risk (process failures)
                - Concentration risk (portfolio impact)
                
                Use statistical models for accurate risk quantification.


                Step 2: Fraud Detection  focused on identifying fraudulent applications:
                1. Analyze application data for inconsistencies
                2. Detect synthetic identity fraud
                3. Identify document manipulation or forgery
                4. Flag suspicious behavioral patterns
                5. Cross-reference against fraud databases
                6. Generate fraud risk scores
                
                Use pattern recognition and anomaly detection techniques.


"""

agent = Agent(
    system_prompt=system_prompt, 
    tools=[],
    name="Risk Analysis Agent",
    description="An agent to do Risk Analysis",
)

host, port = "0.0.0.0", 9000

a2a_server = A2AServer(
    agent=agent,
    http_url=runtime_url,
    serve_at_root=True
)

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "healthy"}

app.mount("/", a2a_server.to_fastapi_app())

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
