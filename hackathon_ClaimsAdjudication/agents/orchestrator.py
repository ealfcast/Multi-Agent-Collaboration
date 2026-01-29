import logging
import json
import asyncio
from typing import Dict, Optional
from urllib.parse import quote
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from helpers.utils import get_cognito_secret, reauthenticate_user, get_ssm_parameter, SSM_APPRAISAL_AGENT_ARN, SSM_SETTLEMENT_AGENT_ARN

from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduced timeouts to prevent hanging
DEFAULT_TIMEOUT = 15  # 15s instead of 300s
AGENT_TIMEOUT = 10    # 10s per agent call

# Global cache and connection pool
_cache = {
    'cognito_config': None,
    'agent_arns': {},
    'agent_cards': {},
    'http_client': None
}

app = BedrockAgentCoreApp()

def get_cached_config():
    """Cache all expensive operations"""
    if not _cache['agent_arns']:
        _cache['agent_arns'] = {
            'docs': get_ssm_parameter(SSM_APPRAISAL_AGENT_ARN),
            'blogs': get_ssm_parameter(SSM_SETTLEMENT_AGENT_ARN)
        }
    
    if not _cache['cognito_config']:
        secret = json.loads(get_cognito_secret())
        _cache['cognito_config'] = {
            'client_id': secret.get("client_id"),
            'client_secret': secret.get("client_secret")
        }
    
    return _cache['agent_arns'], _cache['cognito_config']

def get_bearer_token():
    """Generate fresh bearer token for each request"""
    _, config = get_cached_config()
    return reauthenticate_user(
        config['client_id'], 
        config['client_secret']
    )

def get_http_client():
    """Reuse HTTP client with aggressive timeouts"""
    if not _cache['http_client']:
        _cache['http_client'] = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            http2=True  # Enable HTTP/2 for better performance
        )
    return _cache['http_client']

def create_message(text: str) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
    )

async def send_agent_message(message: str, agent_type: str) -> Optional[str]:
    """Optimized agent communication with circuit breaker pattern"""
    try:
        agent_arns, _ = get_cached_config()
        agent_arn = agent_arns[agent_type]
        bearer_token = get_bearer_token()
        
        from boto3.session import Session
        region = Session().region_name
        
        escaped_arn = quote(agent_arn, safe='')
        runtime_url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{escaped_arn}/invocations/"
        
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id': str(uuid4())
        }
        
        httpx_client = get_http_client()
        httpx_client.headers.update(headers)
        
        # Cache agent card
        if agent_arn not in _cache['agent_cards']:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=runtime_url)
            _cache['agent_cards'][agent_arn] = await asyncio.wait_for(
                resolver.get_agent_card(), timeout=5.0
            )
        
        agent_card = _cache['agent_cards'][agent_arn]
        
        # Create client with non-streaming mode
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        msg = create_message(message)
        
        # Use timeout for the entire operation
        async with asyncio.timeout(AGENT_TIMEOUT):
            async for event in client.send_message(msg):
                if isinstance(event, Message):
                    return event.parts[0].text if event.parts else "No response"
                elif isinstance(event, tuple) and len(event) == 2:
                    return event[0].parts[0].text if event[0].parts else "No response"
        
        return "Timeout: No response received"
        
    except asyncio.TimeoutError:
        logger.warning(f"Timeout calling {agent_type} agent")
        return f"Agent {agent_type} timed out"
    except Exception as e:
        logger.error(f"Error calling {agent_type}: {e}")
        return f"Error: {str(e)[:100]}"
