import os
import logging
import asyncio
from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.multiagent.a2a import A2AServer
from strands.tools.mcp import MCPClient
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
runtime_url = os.environ.get('AGENTCORE_RUNTIME_URL', 'http://127.0.0.1:9000/')
host, port = "0.0.0.0", 9000

# Global MCP client with lazy initialization
_mcp_client = None

async def get_mcp_client():
    """Lazy initialization of MCP client with timeout"""
    global _mcp_client
    if _mcp_client is None:
        try:
            _mcp_client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command="uvx", 
                        args=["awslabs.aws-documentation-mcp-server@latest"]
                    )
                )
            )
            # Start with timeout
            await asyncio.wait_for(_mcp_client.start(), timeout=10.0)
            logger.info("MCP client initialized")
        except asyncio.TimeoutError:
            logger.error("MCP client startup timed out")
            _mcp_client = None
        except Exception as e:
            logger.error(f"MCP client failed: {e}")
            _mcp_client = None
    return _mcp_client

system_prompt = """
                You are the Financial Analysis Manager responsible for comprehensive credit evaluation and income verification.
                Your tasks include:

               Step 1: Credit Analysis
                1. Coordinate credit score analysis with Credit Score Agent
                2. Analyze credit history patterns and trends
                3. Evaluate credit utilization and payment history
                4. Assess credit mix and account age
                5. Identify credit red flags or concerns
                6. Provide consolidated credit assessment summary
                
                Focus Areas:
                - FICO/VantageScore analysis
                - Credit report anomalies
                - Recent credit inquiries
                - Derogatory marks evaluation
                - Credit stability assessment
                
                Provide quantitative scores and qualitative insights for decision-making.

                Step 2: Income Verification
                1. Coordinate income verification through multiple sources
                2. Validate employment status and stability
                3. Verify asset declarations and documentation
                4. Cross-reference financial statements
                5. Identify discrepancies or inconsistencies
                6. Provide comprehensive verification summary
                
                Verification Standards:
                - Income source diversity and stability
                - Employment tenure and position
                - Asset liquidity and ownership
                - Documentation authenticity
                - Financial statement consistency

"""

# Initialize agent with minimal tools first
agent = Agent(
    system_prompt=system_prompt, 
    tools=[],  # Start with no tools, add dynamically
    name="Financial Analysis Agent",
    description="An agent to do Financial Analysis.",
)

# Add tools dynamically when MCP is ready
async def setup_agent_tools():
    """Setup agent tools when MCP client is ready"""
    try:
        mcp_client = await get_mcp_client()
        if mcp_client:
            tools = await asyncio.wait_for(
                mcp_client.list_tools_async(), 
                timeout=5.0
            )
            agent.tools = [tools] if tools else []
            logger.info("Agent tools configured")
    except Exception as e:
        logger.warning(f"Could not setup MCP tools: {e}")

a2a_server = A2AServer(
    agent=agent,
    http_url=runtime_url,
    serve_at_root=True
)

@app.get("/ping")
def ping():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Initialize MCP client on startup"""
    await setup_agent_tools()

app.mount("/", a2a_server.to_fastapi_app())

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
