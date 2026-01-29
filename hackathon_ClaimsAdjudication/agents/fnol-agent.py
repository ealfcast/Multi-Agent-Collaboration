
import os
import logging
import asyncio
from mcp import stdio_client, StdioServerParameters
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools import tool
from strands.multiagent.a2a import A2AServer
from strands.tools.mcp import MCPClient
import argparse
from fastapi import FastAPI
import uvicorn

# Standard library imports
import json
from datetime import datetime

# AWS SDK
import boto3

from typing import Dict, Any, List
from dataclasses import dataclass



print("✓ All dependencies imported successfully")




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AWS Configuration
AWS_REGION = "us-east-1"
MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Initialize Bedrock client to verify connectivity
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

print(f"✓ AWS Bedrock configured for region: {AWS_REGION}")
print(f"✓ Using model: {MODEL_ID}")



# Initialize Bedrock model
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)



# Create FNOL Processing Agent
fnol_agent = Agent(
    name="FNOL Data Extraction Specialist",
    description="Intelligent agent for automated auto insurance claims FNOL processing",
    model=model,
    system_prompt="""You are a Claims Data Extraction Specialist.

ROLE: Process First Notice of Loss (FNOL) forms and extract structured claim data.

INPUTS: Raw FNOL document/form data
OUTPUTS: Structured JSON with validated claim information

INSTRUCTIONS:
1. Extract key data points:
   - Policy number, claim date, incident location, claimant details
   - Incident description, damages reported, witnesses
   - Supporting documentation references

2. Validate data completeness:
   - Flag missing required fields
   - Identify inconsistencies in dates/locations
   - Check format compliance (policy numbers, contact info)

3. Flag potential data quality issues with specific error codes
4. If critical information is missing, generate specific follow-up questions

ERROR HANDLING: If form is illegible or severely incomplete, flag for manual review with detailed reasoning.

QUALITY CHECK: Ensure all extracted monetary amounts, dates, and identifiers are properly formatted.

OUTPUT FORMAT: 
Return the same structured JSON object provided in the input

Return, in a separete section, the list recommendations: Next steps or follow-up actions needed
"""
)

print("FNOL Processing Agent created")



################# A2A ################
app = FastAPI()
runtime_url = os.environ.get('AGENTCORE_RUNTIME_URL', 'http://127.0.0.1:9000/')
host, port = "0.0.0.0", 9000

a2a_server = A2AServer(
    agent=fnol_agent,
    http_url=runtime_url,
    serve_at_root=True,
    
)

@app.get("/ping")
def ping():
    return {"status": "healthy"}

# @app.on_event("startup")
# async def startup_event():
#     """Initialize MCP client on startup"""
#     await setup_agent_tools()

app.mount("/", a2a_server.to_fastapi_app())



if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)

################# A2A ################
