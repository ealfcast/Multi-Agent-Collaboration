import os
import logging
import asyncio
from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.models import BedrockModel
from strands.tools import tool
from strands.multiagent.a2a import A2AServer
from strands.tools.mcp import MCPClient
import argparse
from fastapi import FastAPI
import uvicorn
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent, BeforeModelCallEvent, BeforeToolCallEvent
from pydantic import BaseModel
from botocore.config import Config as BotocoreConfig
from strands.telemetry import StrandsTelemetry
from findings_utils import extract_reasoning_findings
from strands_tools import retrieve

# Standard library imports
import json
from datetime import datetime

# AWS SDK
import boto3




# Configure the root strands logger
# logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add a handler to see the logs
# logging.basicConfig(
#     format="%(levelname)s | %(name)s | %(message)s", 
#     handlers=[logging.StreamHandler()]
# )

# Setup tracing - commented out for now as this adds a lot of trace output that really isn't interesting
StrandsTelemetry().setup_console_exporter()

# NOTE: To send the OTEL data to an ADOT collector, additional exporter needs to be used





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
