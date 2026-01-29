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

from ddgs import DDGS
from ddgs.exceptions import RatelimitException, DDGSException



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

# Supply the pre-installed polciy and guardrail IDs
ARC_POLICY_ARN = "arn:aws:bedrock:us-east-1:161615149547:automated-reasoning-policy/malxiyr0ojy2"
GUARDRAIL_ID = "an852wptcjol"
GUARDRAIL_VERSION = "4"
KNOWLEDGE_BASE_ID = "CZDJXI9C4E"
# NOTE: the default model for Strands is us.anthropic.claude-sonnet-4-20250514-v1:0
# MODEL_ID = "us.amazon.nova-lite-v1:0"
# MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


# Setup the environment for the agent and tool
# Allow for the metadata to be retrieved on sources from the KB
os.environ['RETRIEVE_ENABLE_METADATA_DEFAULT'] = 'true'
# Allow for the retrieve tool to interact with the KB
os.environ['KNOWLEDGE_BASE_ID'] = KNOWLEDGE_BASE_ID



# Define a notification hook to listen to events and then process the result and call
# Automated Reasoning attached via the Guardrail and report on the findings.  This
# can be used possibly re-write the output or add a flag on if the output is correct.
class NotifyOnlyGuardrailsHook(HookProvider):
    
    def __init__(self, guardrail_id: str, guardrail_version: str, arc_policy_arn: str):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.arc_policy_arn = arc_policy_arn
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.input = ''
        self.claim_valid = True
        self.findings = ''
        self.policy_definition = {}
        self.before_tool_event_flag = False
        self.before_model_event_flag = False

        if self.arc_policy_arn:
            try:
                bedrock_client = boto3.client('bedrock')
                response = bedrock_client.export_automated_reasoning_policy_version(policyArn=self.arc_policy_arn)
                self.policy_definition = response.get('policyDefinition', {})
            except Exception as e:
                print(f"Error getting policy definition: {str(e)}")
                raise

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeModelCallEvent, self.before_model_event)
        registry.add_callback(BeforeToolCallEvent, self.before_tool_event)
        registry.add_callback(MessageAddedEvent, self.message_added)

    def message_added(self, event: MessageAddedEvent) -> None:
        if self.before_tool_event_flag:
            # Since a tool was called, just ignore this message addition
            self.before_tool_event_flag = False
            return
        
        # Get the content
        content = "".join(block.get("text", "") for block in event.message.get("content", []))

        # Determine the source
        if event.message.get("role") == "user":
            # Store the input for later usage and allow the loop to continue to process
            self.input = content
            return

        if not content:
            return
            #do something 

        # Capture if this is the first time that findings will be created
        first_findings = (not self.findings)

        # Format a request to send to the guardrail
        content_to_validate = [
            {"text": {"text": self.input, "qualifiers": ["query"]}},
            {"text": {"text": content, "qualifiers": ["guard_content"]}}
        ]
        print ("HERE LOOKIE HERE",content_to_validate)
        
        # Call the guardrail
        response = self.bedrock_client.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source="OUTPUT",
            content=content_to_validate
        )

        # Determine if the output is correct
        self.findings = extract_reasoning_findings(response, self.policy_definition)
         
        assessments = response.get("assessments", [])
        if assessments and len(assessments):
            self.claim_valid = False

        # Add information to the output
        if self.findings and first_findings:
            new_output = contentbedrock_model
            new_output = new_output + f"\n\nfindings: {self.findings}"
            new_output = new_output + f"\n\nclaim_valid: {self.claim_valid}"
            event.message["content"][0]["text"] = new_output
        
    def before_model_event(self, event: BeforeModelCallEvent) -> None:
        self.before_model_event_flag = True

    def before_tool_event(self, event: BeforeToolCallEvent) -> None:
        self.before_tool_event_flag = True

# Create structured output
class StructuredOutputModel(BaseModel):
    claim_valid: bool
    content: str
    findings: str

# Provide the config for botocore
boto_config = BotocoreConfig(
    retries={"max_attempts": 3, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60
)

# Create a Bedrock model with guardrail configuration
bedrock_model = BedrockModel(
    boto_client_config=boto_config,
    model_id=MODEL_ID,
    # NOTE: An alternative option is to supply the guardrail here.  If going that route, the ARc findings aren't present.
    # To ensure that the findings are present and can be used to re-write the output, rely on a hook
)

agent_instructions="""You are an expert automotive claims validaiton specialist that determines if the users auto insurance claim is valid based on the provided information and details within the policy contract.
    
You will be provided with JSON data that has claim information and vehicle damage information, you should:
1. Extract from the JSON data required claim information to be validated
2. Focus on time of event and time of claim creation
3. Focus on claims and coverage inconsistencies

Your responses should :
- If your response is "Valid claim", then output the provide the full JSON structure provided from the input
- If you response is "Invalid claim", then output the response "This Claim is Invalid and no appraisal nor settlement is required"
- If you response is "Invalid claim", then provide clear explanation on why is invalid in the output
- In cases where a clear outcome is not present, recommend the user to check with their insurance agent directly. 

Take your time to think though the answer and evalute carefully."

"""


# Create agent with the guardrail-protected model
agent = Agent(
    name="Claims Validator - ARC",
    description="A Single agent with Claims Validation tools capabilities",
    model=bedrock_model,
    hooks=[NotifyOnlyGuardrailsHook(GUARDRAIL_ID, GUARDRAIL_VERSION, ARC_POLICY_ARN)],
    tools=[retrieve],
    system_prompt=agent_instructions
)



################# A2A ################
app = FastAPI()
runtime_url = os.environ.get('AGENTCORE_RUNTIME_URL', 'http://127.0.0.1:9000/')
host, port = "0.0.0.0", 9000

a2a_server = A2AServer(
    agent=agent,
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
