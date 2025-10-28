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

# Standard library imports
import json
from datetime import datetime

# AWS SDK
import boto3


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


#Vehicle Type Classification

def classify_vehicle_type(make: str, model: str) -> str:
    """
    Classify vehicle into type category for pricing purposes.
    
    Args:
        make: Vehicle manufacturer (e.g., "Honda", "Ford")
        model: Vehicle model name (e.g., "Accord", "F-150")
        
    Returns:
        One of: "Car", "Truck", "SUV", "Van"
    """
    # Normalize inputs for case-insensitive matching
    model_lower = model.lower()
    
    # SUV classification - common SUV models
    suv_keywords = [
        'explorer', 'pilot', 'cr-v', 'crv', 'rav4', 'highlander',
        'tahoe', 'suburban', 'expedition', '4runner', 'pathfinder',
        'traverse', 'durango', 'grand cherokee', 'cherokee', 'wrangler',
        'rogue', 'murano', 'armada', 'sequoia', 'land cruiser',
        'cx-5', 'cx-9', 'outback', 'forester', 'ascent',
        'santa fe', 'tucson', 'palisade', 'telluride', 'sorento',
        'sportage', 'atlas', 'tiguan', 'touareg', 'x5', 'x3',
        'q5', 'q7', 'gx', 'rx', 'nx', 'xt5', 'xt6', 'escalade'
    ]
    
    # Truck classification - pickup trucks
    truck_keywords = [
        'f-150', 'f150', 'f-250', 'f250', 'f-350', 'f350',
        'silverado', 'sierra', 'ram', 'tundra', 'tacoma',
        'ranger', 'colorado', 'canyon', 'frontier', 'titan',
        'ridgeline', 'gladiator', 'maverick'
    ]
    
    # Van classification - minivans and cargo vans
    van_keywords = [
        'odyssey', 'sienna', 'pacifica', 'caravan', 'grand caravan',
        'transit', 'sprinter', 'promaster', 'metris', 'express',
        'savana', 'nv', 'quest', 'sedona', 'carnival'
    ]
    
    # Check for SUV
    for keyword in suv_keywords:
        if keyword in model_lower:
            return "SUV"
    
    # Check for Truck
    for keyword in truck_keywords:
        if keyword in model_lower:
            return "Truck"
    
    # Check for Van
    for keyword in van_keywords:
        if keyword in model_lower:
            return "Van"
    
    # Default to Car for sedans, coupes, hatchbacks, and unknown models
    return "Car"

# Pricing matrix: Component costs by vehicle type
# Format: {component: {vehicle_type: (min_cost, max_cost)}}
PRICING_MATRIX = {
    # Bumpers - Front and rear impact protection
    "front bumper": {
        "Car": (800, 1500),
        "Truck": (1200, 2000),
        "SUV": (1000, 1800),
        "Van": (900, 1600)
    },
    "rear bumper": {
        "Car": (800, 1500),
        "Truck": (1200, 2000),
        "SUV": (1000, 1800),
        "Van": (900, 1600)
    },
    
    # Hood - Engine compartment cover
    "hood": {
        "Car": (1000, 2000),
        "Truck": (1500, 3000),
        "SUV": (1200, 2500),
        "Van": (1100, 2200)
    },
    
    # Trunk - Rear storage compartment
    "trunk": {
        "Car": (1200, 2500),
        "Truck": (1800, 3500),
        "SUV": (1500, 3000),
        "Van": (1400, 2800)
    },
    
    # Doors - Per door pricing
    "door": {
        "Car": (1500, 3000),
        "Truck": (2000, 4000),
        "SUV": (1800, 3500),
        "Van": (1700, 3200)
    },
    
    # Side panels - Per panel pricing
    "side panel": {
        "Car": (1800, 3500),
        "Truck": (2500, 5000),
        "SUV": (2200, 4500),
        "Van": (2000, 4000)
    },
    
    # Fenders - Per fender pricing
    "fender": {
        "Car": (900, 2000),
        "Truck": (1400, 2800),
        "SUV": (1200, 2500),
        "Van": (1000, 2200)
    },
    
    # Headlights - Per light pricing
    "headlight": {
        "Car": (400, 800),
        "Truck": (600, 1200),
        "SUV": (500, 1000),
        "Van": (450, 900)
    },
    
    # Taillights - Per light pricing
    "taillight": {
        "Car": (300, 600),
        "Truck": (500, 800),
        "SUV": (400, 700),
        "Van": (350, 650)
    },
    
    # Grille - Front air intake
    "grille": {
        "Car": (400, 900),
        "Truck": (700, 1500),
        "SUV": (600, 1200),
        "Van": (500, 1000)
    },
    
    # Radiator - Cooling system
    "radiator": {
        "Car": (800, 1500),
        "Truck": (1200, 2500),
        "SUV": (1000, 2000),
        "Van": (900, 1800)
    },
    
    # Frame damage - Structural damage (most expensive)
    "frame": {
        "Car": (2000, 5000),
        "Truck": (3000, 8000),
        "SUV": (2500, 6500),
        "Van": (2200, 6000)
    }
}


def get_component_cost(component: str, vehicle_type: str) -> float:
    """
    Get the estimated cost for a damaged component based on vehicle type.
    Uses the midpoint of the cost range for the estimate.
    
    Args:
        component: Name of damaged component (e.g., "rear bumper", "hood")
        vehicle_type: Vehicle classification ("Car", "Truck", "SUV", "Van")
    
    Returns:
        Estimated cost as float (midpoint of range)
    """
    # Normalize component name to lowercase for matching
    component_lower = component.lower().strip()
    
    # Handle common variations in component naming
    component_mapping = {
        "front bumper": "front bumper",
        "rear bumper": "rear bumper",
        "bumper": "rear bumper",  # Default to rear if not specified
        "hood": "hood",
        "trunk": "trunk",
        "door": "door",
        "doors": "door",
        "side panel": "side panel",
        "panel": "side panel",
        "fender": "fender",
        "fenders": "fender",
        "headlight": "headlight",
        "headlights": "headlight",
        "taillight": "taillight",
        "taillights": "taillight",
        "tail light": "taillight",
        "tail lights": "taillight",
        "grille": "grille",
        "grill": "grille",
        "radiator": "radiator",
        "frame": "frame",
        "frame damage": "frame"
    }
    
    # Map component to standard name
    standard_component = component_mapping.get(component_lower, component_lower)
    
    # Look up pricing
    if standard_component in PRICING_MATRIX:
        min_cost, max_cost = PRICING_MATRIX[standard_component][vehicle_type]
        # Return midpoint of range
        return (min_cost + max_cost) / 2
    
    # If component not found, return 0 (will be handled in error reporting)
    return 0.0


#Repair Cost Estimation Tool
@tool
def estimate_repair_costs(
    vehicle_year: int,
    vehicle_make: str,
    vehicle_model: str,
    vehicle_vin: str,
    vehicle_mileage: int,
    damage_areas: list,
    damage_description: str,
    claim_number: str
) -> str:
    """
    Estimate repair costs for a collision-damaged vehicle based on vehicle type,
    damaged components, and applicable cost multipliers.
    
    This tool analyzes vehicle information and damage details to generate a comprehensive
    repair cost estimate. It classifies the vehicle into a pricing tier (Car, Truck, SUV, Van),
    calculates costs for each damaged component, applies multipliers for luxury brands and
    vehicle age, and returns a detailed JSON estimate.
    
    Args:
        vehicle_year: Year of vehicle manufacture (e.g., 2020)
        vehicle_make: Vehicle manufacturer name (e.g., "Honda", "Ford", "BMW")
        vehicle_model: Vehicle model name (e.g., "Accord", "F-150", "X5")
        vehicle_vin: Vehicle Identification Number (17-character alphanumeric)
        vehicle_mileage: Current vehicle mileage in miles (e.g., 35650)
        damage_areas: List of damaged components (e.g., ["Rear bumper", "Trunk", "Taillights"])
        damage_description: Detailed description of damage (e.g., "Rear bumper dented and paint scratched")
        claim_number: Insurance claim number for report metadata (e.g., "CL-2023-1156789")
    
    Returns:
        JSON string containing:
        - estimate_metadata: Claim number, estimate date, disclaimer
        - vehicle_information: Year, make, model, VIN, mileage, vehicle type classification
        - damage_assessment: Array of damaged components with individual costs and subtotal
        - cost_adjustments: Applied multipliers (luxury brand, age discount) with reasons
        - total_estimate: Final estimated repair cost in USD
    
    Example:
        >>> estimate = estimate_repair_costs(
        ...     vehicle_year=2020,
        ...     vehicle_make="Honda",
        ...     vehicle_model="Accord",
        ...     vehicle_vin="1HGCV2F35LA007149",
        ...     vehicle_mileage=35650,
        ...     damage_areas=["Rear bumper", "Trunk", "Taillights"],
        ...     damage_description="Rear-end collision damage",
        ...     claim_number="CL-2023-1156789"
        ... )
    """
    try:
        # Validate required parameters
        missing_fields = []
        if not vehicle_year:
            missing_fields.append("vehicle_year")
        if not vehicle_make:
            missing_fields.append("vehicle_make")
        if not vehicle_model:
            missing_fields.append("vehicle_model")
        if not vehicle_vin:
            missing_fields.append("vehicle_vin")
        if not damage_areas or len(damage_areas) == 0:
            missing_fields.append("damage_areas")
        if not claim_number:
            missing_fields.append("claim_number")
        
        if missing_fields:
            return json.dumps({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
                "status": "validation_failed"
            })


        #Vehicle Type Classification
        #Classify the vehicle into one of four standard categories for pricing purposes:
        #Car: Sedans, coupes, hatchbacks (default category)
        #Truck: Pickup trucks (F-150, Silverado, Ram, etc.)
        #SUV: Sport utility vehicles (Explorer, Pilot, CR-V, RAV4, etc.)
        #Van: Minivans and cargo vans (Odyssey, Sienna, Transit, etc.)
        #The classification determines the base pricing tier for repair cost estimation.
    
        # Classify vehicle type for pricing
        vehicle_type = classify_vehicle_type(vehicle_make, vehicle_model)
        
        # Calculate cost multipliers
        multipliers = []
        cumulative_multiplier = 1.0
        
        # Luxury brand multiplier (1.3x)
        luxury_brands = [
            "lexus", "bmw", "mercedes-benz", "mercedes", "audi", "porsche",
            "jaguar", "land rover", "cadillac", "lincoln", "acura",
            "infiniti", "genesis"
        ]
        
        if vehicle_make.lower() in luxury_brands:
            luxury_multiplier = 1.3
            cumulative_multiplier *= luxury_multiplier
            multipliers.append({
                "type": "luxury_brand",
                "factor": luxury_multiplier,
                "reason": f"{vehicle_make} is a luxury brand with higher parts and labor costs"
            })
        
        # Age discount multiplier (0.85x for vehicles older than 10 years)
        current_year = datetime.now().year
        vehicle_age = current_year - vehicle_year
        
        if vehicle_age > 10:
            age_multiplier = 0.85
            cumulative_multiplier *= age_multiplier
            multipliers.append({
                "type": "age_discount",
                "factor": age_multiplier,
                "reason": f"Vehicle is {vehicle_age} years old (older than 10 years), parts costs are lower"
            })
        
        # Calculate costs for each damaged component
        damaged_components = []
        subtotal = 0.0
        
        for component in damage_areas:
            base_cost = get_component_cost(component, vehicle_type)
            
            if base_cost == 0.0:
                # Component not found in pricing matrix, skip it
                continue
            
            # Apply multipliers to component cost
            adjusted_cost = base_cost * cumulative_multiplier
            subtotal += adjusted_cost
            
            damaged_components.append({
                "component": component,
                "damage_description": damage_description,
                "estimated_cost": round(adjusted_cost, 2)
            })
        
        # Calculate adjustment total (difference from base subtotal)
        base_subtotal = subtotal / cumulative_multiplier if cumulative_multiplier != 1.0 else subtotal
        adjustment_total = subtotal - base_subtotal
        
        # Generate JSON output
        estimate = {
            "estimate_metadata": {
                "claim_number": claim_number,
                "estimate_date": datetime.now().isoformat(),
                "disclaimer": "This is a preliminary estimate. Actual repair costs may vary based on hidden damage, parts availability, and labor rates."
            },
            "vehicle_information": {
                "year": vehicle_year,
                "make": vehicle_make,
                "model": vehicle_model,
                "vin": vehicle_vin,
                "mileage": vehicle_mileage,
                "vehicle_type": vehicle_type
            },
            "damage_assessment": {
                "damaged_components": damaged_components,
                "subtotal": round(subtotal, 2)
            },
            "cost_adjustments": {
                "multipliers": multipliers,
                "adjustment_total": round(adjustment_total, 2)
            },
            "total_estimate": {
                "amount": round(subtotal, 2),
                "currency": "USD"
            }
        }
        
        return json.dumps(estimate, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": "Cost calculation failed",
            "details": str(e),
            "status": "calculation_failed"
        })



# Initialize Bedrock model
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

#Agent Instructions

agent_instructions="""You are an expert automotive claims adjuster specializing in repair cost estimation.
    
When provided with vehicle damage information, you should:
1. Use the estimate_repair_costs tool to generate detailed cost estimates and return JSON estimate
2. Analyze the returned JSON estimate carefully
3. Provide a clear, professional summary of the repair costs
4. Highlight key factors affecting the estimate (vehicle type, luxury brand, age, etc.)
5. Explain the breakdown of costs by damaged component
6. Note any cost adjustments and their reasons
7. Present the total estimated repair cost prominently

Your responses should be:
- Professional and authoritative
- Clear and easy to understand for insurance adjusters
- Detailed enough to justify the estimate
- Transparent about the preliminary nature of the estimate
- Include a section to provide the full JSON estimate

Always remind users that this is a preliminary estimate and actual costs may vary based on hidden damage, parts availability, and labor rates.
"""

# Create agent with repair cost estimation tool
agent = Agent(
    name="Repair Cost Estimator",
    description="A Single agent wit Appraisal tools capabilities",
    model=model,
    tools=[estimate_repair_costs],
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

################# Normal Config ################

# def strands_agent_bedrock(payload):
#     """
#     Invoke the agent with a payload
#     """
#     user_input = payload.get("prompt")
#     print (user_input)
#     response = agent(user_input)
#     return response.message['content'][0]['text']

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("payload", type=str)
#     args = parser.parse_args()
#     response = strands_agent_bedrock(json.loads(args.payload))

################# Normal Config ################

