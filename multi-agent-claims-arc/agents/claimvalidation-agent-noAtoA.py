
"""
CLaims Validation — Strands agent deployed on Bedrock AgentCore Runtime.

Tools ( RAG and ARC Hooks):
"""
import os
import logging
import asyncio
import boto3
import json
import uvicorn
from datetime import datetime
from strands.models import BedrockModel
from strands.tools import tool
from strands.multiagent.a2a import A2AServer
from fastapi import FastAPI
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent, BeforeModelCallEvent, BeforeToolCallEvent
from pydantic import BaseModel
from botocore.config import Config as BotocoreConfig
from strands.telemetry import StrandsTelemetry
from findings_utils import extract_reasoning_findings
from strands_tools import retrieve
import re
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent, tool

# Configure the root strands logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a handler to see the logs
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)

app = BedrockAgentCoreApp()

# Setup tracing - commented out for now as this adds a lot of trace output that really isn't interesting
# NOTE: To send the OTEL data to an ADOT collector, additional exporter needs to be used
StrandsTelemetry().setup_console_exporter()

# The default values as pulled fomr the environment.  These values
# are replaced after the file is written out.
BEDROCK_REGION = 'us-east-1'
ARC_POLICY_ARN = 'arn:aws:bedrock:us-east-1:161615149547:automated-reasoning-policy/6if243lxvpvp:1'
GUARDRAIL_ID = 'tqiexl44264z'
GUARDRAIL_VERSION = 'DRAFT'
KNOWLEDGE_BASE_ID = 'KOE1M5ZOAU'

# Initialize Bedrock client to verify connectivity
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=BEDROCK_REGION
)

print(f"✓ AWS Bedrock configured for region: {BEDROCK_REGION}")

# Setup the environment for the agent and tool
# Allow for the metadata to be retrieved on sources from the KB
os.environ['RETRIEVE_ENABLE_METADATA_DEFAULT'] = 'true'

# Store the Knowledge Base ID in the environment to allow the tool
# to interact with the KB
os.environ['KNOWLEDGE_BASE_ID'] = KNOWLEDGE_BASE_ID

# Define a notification hook to listen to events and then process the result and call
# Automated Reasoning attached via the Guardrail and report on the findings.  This
# can be used possibly re-write the output or add a flag on if the output is correct.
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
            new_output = content
            new_output = new_output + f"\n*** FINDINGS: ***:\n{self.findings}"
            new_output = new_output + f"\n*** CLAIM VALID: ***:\n{self.claim_valid}"
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

Create a section called "Claims Policy Validation Explainability". Use the Automated Reasoning Checks rules to provide explanation and list the rules that were valid and not valid
Provide the following for this section:
he Automated Reasoning Checks rules to provide explanation and list the rules that were valid and not validz
## Finding 
**Finding Type:** 

### Translation:
#### Premises:
#### Claims:
#### Untranslated Claims:

**Confidence Score:** 

### Claims True Scenario:


Take your time to think though the answer and evalute carefully."

"""

# Create agent with the guardrail-protected model
myagent = Agent(
    name="Claims Validator - ARC",
    description="A Single agent with ARC Claims Validation tools capabilities",
    hooks=[NotifyOnlyGuardrailsHook(GUARDRAIL_ID, GUARDRAIL_VERSION, ARC_POLICY_ARN)],
    tools=[retrieve],
    system_prompt=agent_instructions
)


# ----------------------------



# Session cache: session_id -> Agent (preserves conversation history across turns)
_SESSION_AGENTS: dict[str, Agent] = {}


@app.entrypoint
async def invoke(payload, context):
    """Handle an agent invocation from AgentCore Runtime."""
    prompt = payload.get("prompt", "")
    session_id = context.session_id
    logger.info("Received prompt (session=%s): %s", session_id, prompt[:80])

    if session_id and session_id in _SESSION_AGENTS:
        agent = _SESSION_AGENTS[session_id]
    else:
        # agent = Agent(model=_MODEL, tools=_TOOLS, system_prompt=SYSTEM_PROMPT)
        agent = myagent
        if session_id:
            _SESSION_AGENTS[session_id] = agent

    parts = []
    async for event in agent.stream_async(prompt):
        if "data" in event:
            parts.append(str(event["data"]))
    response = "".join(parts)
    # Strip inline <thinking>...</thinking> blocks so spans contain only the final answer
    response = re.sub(
        r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL
    ).strip()
    return response


if __name__ == "__main__":
    app.run()
