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


@dataclass
class CoveragePolicy:
    """Policy coverage configuration"""
    collision_coverage: float = 50000.0
    deductible: float = 500.0
    policy_number: str = "POL-2023-456789"
    policy_status: str = "ACTIVE"
    coverage_type: str = "COMPREHENSIVE"


class AutoInsuranceClaimsAgent:
    """
    AWS Strands Agent for Auto Insurance Claims Settlement Automation
    Handles complete claim processing from verification to settlement
    """
    
    def __init__(self):
        self.policy_database = {
            "CL-2023-1156789": CoveragePolicy(
                collision_coverage=50000.0,
                deductible=500.0,
                policy_number="POL-2023-456789",
                policy_status="ACTIVE",
                coverage_type="COMPREHENSIVE"
            )
        }
        self.claim_data = None
        self.coverage_info = None
        self.settlement_details = {}
    
    @tool
    def verify_coverage_limits(self, claim_number: str, estimated_cost: float) -> Dict[str, Any]:
        """
        Verifies applicable coverage limits against policy terms.
        
        Args:
            claim_number: Unique claim identifier
            estimated_cost: Total estimated repair cost
            
        Returns:
            Coverage verification details with policy limits and applicability
        """
        policy = self.policy_database.get(claim_number)
        
        if not policy:
            return {
                "status": "ERROR",
                "message": f"No policy found for claim {claim_number}",
                "coverage_applicable": False
            }
        
        if policy.policy_status != "ACTIVE":
            return {
                "status": "DENIED",
                "message": f"Policy status is {policy.policy_status}",
                "coverage_applicable": False,
                "policy_citation": f"Policy {policy.policy_number} - Section 2.1: Coverage requires active policy status"
            }
        
        coverage_applicable = estimated_cost <= policy.collision_coverage
        
        coverage_result = {
            "status": "VERIFIED",
            "coverage_applicable": coverage_applicable,
            "policy_number": policy.policy_number,
            "coverage_type": policy.coverage_type,
            "coverage_limit": policy.collision_coverage,
            "estimated_cost": estimated_cost,
            "deductible": policy.deductible,
            "within_limits": coverage_applicable,
            "excess_amount": max(0, estimated_cost - policy.collision_coverage)
        }
        
        self.coverage_info = coverage_result
        return coverage_result
    
    @tool
    def apply_deductible_adjustments(self, estimated_cost: float, deductible: float) -> Dict[str, Any]:
        """
        Applies policy deductible to calculate net insurance payout.
        
        Args:
            estimated_cost: Total repair estimate
            deductible: Policy deductible amount
            
        Returns:
            Detailed breakdown of adjusted amounts
        """
        adjusted_amount = max(0, estimated_cost - deductible)
        
        adjustment_details = {
            "original_estimate": estimated_cost,
            "deductible_amount": deductible,
            "adjusted_payout": adjusted_amount,
            "policyholder_responsibility": min(estimated_cost, deductible),
            "insurance_responsibility": adjusted_amount,
            "adjustment_applied": True
        }
        
        self.settlement_details.update(adjustment_details)
        return adjustment_details
    
    @tool
    def evaluate_claim_decision(self, claim_data: Dict[str, Any], coverage_info: Dict[str, Any], 
                                adjusted_amount: float) -> Dict[str, Any]:
        """
        Makes claim decision: APPROVE, DENY, or INVESTIGATE based on policy rules and risk factors.
        
        Args:
            claim_data: Complete claim information
            coverage_info: Coverage verification results
            adjusted_amount: Net payout after deductible
            
        Returns:
            Decision outcome with reasoning and conditions
        """
        decision = {
            "claim_number": claim_data.get("estimate_metadata", {}).get("claim_number"),
            "decision_date": datetime.now().isoformat(),
            "decision_by": "AUTO_CLAIMS_SETTLEMENT_AGENT"
        }
        
        investigation_triggers = []
        
        # Check for high-risk damage indicators
        damage_components = claim_data.get("damage_assessment", {}).get("damaged_components", [])
        for component in damage_components:
            damage_desc = component.get("damage_description", "").lower()
            if "frame damage" in damage_desc or "structural" in damage_desc:
                investigation_triggers.append(
                    f"Possible frame/structural damage detected in {component.get('component')} - requires certified inspection"
                )
        
        # Check auto-approval threshold
        if adjusted_amount > 5000:
            investigation_triggers.append(
                f"Claim amount \${adjusted_amount:,.2f} exceeds auto-approval threshold of \$5,000"
            )
        
        # DENY: Coverage not applicable
        if not coverage_info.get("coverage_applicable", False):
            decision["outcome"] = "DENY"
            decision["reasoning"] = "Claim amount exceeds maximum policy coverage limits"
            decision["policy_citations"] = [
                f"Policy {coverage_info.get('policy_number')} - Section 4.2: Collision coverage limit is \${coverage_info.get('coverage_limit'):,.2f}",
                f"Claim estimate \${coverage_info.get('estimated_cost'):,.2f} exceeds coverage by \${coverage_info.get('excess_amount'):,.2f}"
            ]
            decision["details"] = "The estimated repair costs exceed the maximum coverage limit specified in your policy. You may appeal this decision with additional documentation."
            return decision
        
        # INVESTIGATE: Risk factors present
        if investigation_triggers:
            decision["outcome"] = "INVESTIGATE"
            decision["reasoning"] = "Claim requires additional investigation before settlement authorization"
            decision["investigation_requirements"] = investigation_triggers
            decision["required_actions"] = [
                "Schedule comprehensive vehicle inspection by certified adjuster",
                "Obtain structural integrity assessment from authorized facility",
                "Document all hidden damage through detailed photographic evidence",
                "Verify repair estimates from manufacturer-certified repair center",
                "Review vehicle history for pre-existing damage"
            ]
            decision["estimated_investigation_timeline"] = "3-5 business days"
            decision["next_steps"] = "Our claims adjuster will contact you within 24 hours to schedule inspection"
            return decision
        
        # APPROVE: All criteria met
        decision["outcome"] = "APPROVE"
        decision["reasoning"] = "Claim meets all policy criteria and approval thresholds"
        decision["approval_conditions"] = [
            "Payment subject to submission of final itemized repair invoice",
            "Repairs must be completed at insurance-approved or certified repair facility",
            "Policyholder responsible for deductible payment directly to repair facility",
            "Final quality inspection may be required before claim closure",
            "Supplemental claims for hidden damage must be submitted within 30 days of initial repair"
        ]
        decision["approved_amount"] = adjusted_amount
        decision["payment_method"] = "Direct deposit to policyholder or two-party check to policyholder and repair facility"
        
        return decision
    
    @tool
    def create_claim_documentation(self, claim_data: Dict[str, Any], coverage_info: Dict[str, Any],
                                   adjustment_details: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates comprehensive audit documentation for regulatory compliance and future reference.
        
        Args:
            claim_data: Original claim submission
            coverage_info: Coverage verification results
            adjustment_details: Financial adjustments
            decision: Claim decision details
            
        Returns:
            Complete documentation package with audit trail
        """
        metadata = claim_data.get("estimate_metadata", {})
        vehicle_info = claim_data.get("vehicle_information", {})
        damage_assessment = claim_data.get("damage_assessment", {})
        
        documentation = {
            "document_type": "AUTO_INSURANCE_CLAIM_SETTLEMENT_RECORD",
            "document_id": f"DOC-{metadata.get('claim_number')}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "regulatory_compliance": "DOI-2023-Standards-Compliant",
            
            "claim_summary": {
                "claim_number": metadata.get("claim_number"),
                "claim_date": metadata.get("estimate_date"),
                "estimate_disclaimer": metadata.get("disclaimer"),
                "vehicle": {
                    "year": vehicle_info.get("year"),
                    "make": vehicle_info.get("make"),
                    "model": vehicle_info.get("model"),
                    "vin": vehicle_info.get("vin"),
                    "mileage": vehicle_info.get("mileage"),
                    "type": vehicle_info.get("vehicle_type")
                },
                "damage_summary": {
                    "total_components_damaged": len(damage_assessment.get("damaged_components", [])),
                    "components": [
                        {
                            "component": comp.get("component"),
                            "description": comp.get("damage_description"),
                            "estimated_cost": comp.get("estimated_cost")
                        }
                        for comp in damage_assessment.get("damaged_components", [])
                    ],
                    "damage_subtotal": damage_assessment.get("subtotal"),
                    "total_estimate": claim_data.get("total_estimate", {}).get("amount")
                }
            },
            
            "coverage_verification": coverage_info,
            "financial_breakdown": adjustment_details,
            "decision_record": decision,
            
            "audit_trail": {
                "processed_by": "AWS_STRANDS_AUTO_CLAIMS_AGENT_v1.0",
                "processing_timestamp": datetime.now().isoformat(),
                "automated_decision": True,
                "verification_steps_completed": [
                    "✓ Policy status and coverage limits verified",
                    "✓ Deductible adjustments calculated and applied",
                    "✓ Risk assessment completed",
                    "✓ Decision matrix evaluation performed",
                    "✓ Compliance documentation generated"
                ],
                "data_sources": ["Policy Database", "Claim Intake System", "Risk Assessment Engine"]
            }
        }
        
        return documentation
    
    @tool
    def generate_settlement_output(self, documentation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates formatted settlement summary with all required output components.
        
        Args:
            documentation: Complete claim documentation
            
        Returns:
            Structured settlement output ready for disbursement processing
        """
        decision = documentation.get("decision_record", {})
        financial = documentation.get("financial_breakdown", {})
        claim_summary = documentation.get("claim_summary", {})
        coverage = documentation.get("coverage_verification", {})
        
        output = {
            "SETTLEMENT_SUMMARY": {
                "claim_number": claim_summary.get("claim_number"),
                "decision": decision.get("outcome"),
                "decision_date": decision.get("decision_date"),
                "vehicle": f"{claim_summary['vehicle']['year']} {claim_summary['vehicle']['make']} {claim_summary['vehicle']['model']}",
                "vin": claim_summary['vehicle']['vin'],
                
                "line_item_breakdown": [
                    {
                        "item": comp.get("component"),
                        "damage_description": comp.get("description"),
                        "estimated_cost": f"\${comp.get('estimated_cost'):,.2f}"
                    }
                    for comp in claim_summary.get("damage_summary", {}).get("components", [])
                ],
                
                "financial_summary": {
                    "total_damage_estimate": f"\${financial.get('original_estimate', 0):,.2f}",
                    "policy_deductible": f"\${financial.get('deductible_amount', 0):,.2f}",
                    "net_insurance_payout": f"\${financial.get('adjusted_payout', 0):,.2f}",
                    "policyholder_responsibility": f"\${financial.get('policyholder_responsibility', 0):,.2f}"
                },
                
                "coverage_details": {
                    "policy_number": coverage.get("policy_number"),
                    "coverage_type": coverage.get("coverage_type"),
                    "coverage_limit": f"\${coverage.get('coverage_limit', 0):,.2f}"
                }
            },
            
            "PAYMENT_AUTHORIZATION": self._generate_payment_authorization(decision, financial, claim_summary),
            
            "REQUIRED_DOCUMENTATION": {
                "mandatory_documents": self._generate_required_docs(decision),
                "submission_method": "Upload via policyholder portal or email to claims@insurance.com",
                "submission_deadline": "Within 30 days of claim approval"
            },
            
            "CONDITIONS_AND_REQUIREMENTS": {
                "decision_specific": decision.get("approval_conditions", 
                                                 decision.get("investigation_requirements", 
                                                            decision.get("policy_citations", []))),
                "general_terms": [
                    "All repairs must be completed within 90 days of approval",
                    "Salvage or total loss determination may supersede this estimate",
                    "Depreciation may apply to parts over 3 years old per policy terms"
                ]
            },
            
            "NEXT_STEPS": self._generate_next_steps(decision),
            
            "CONTACT_INFORMATION": {
                "claims_hotline": "1-800-CLAIMS-1",
                "email": "claims@insurance.com",
                "claims_adjuster": "Will be assigned within 24 hours",
                "online_portal": "https://claims.insurance.com"
            }
        }
        
        return output
    
    def _generate_payment_authorization(self, decision: Dict, financial: Dict, claim_summary: Dict) -> Dict[str, Any]:
        """Generate payment authorization details based on decision outcome"""
        if decision.get("outcome") == "APPROVE":
            return {
                "authorization_status": "✓ AUTHORIZED",
                "authorized_amount": f"\${financial.get('adjusted_payout', 0):,.2f}",
                "authorization_code": f"AUTH-{claim_summary.get('claim_number')}-{datetime.now().strftime('%Y%m%d')}",
                "payment_method": decision.get("payment_method"),
                "disbursement_timeline": "5-7 business days after receipt of required documentation",
                "payee_options": [
                    "Direct deposit to policyholder bank account",
                    "Two-party check (policyholder and repair facility)",
                    "Direct payment to approved repair facility"
                ],
                "authorization_valid_until": "90 days from authorization date"
            }
        elif decision.get("outcome") == "INVESTIGATE":
            return {
                "authorization_status": "⚠ PENDING INVESTIGATION",
                "authorized_amount": "\$0.00 (Pending)",
                "status": "Payment authorization withheld pending investigation completion",
                "reason": decision.get("reasoning"),
                "estimated_resolution": decision.get("estimated_investigation_timeline"),
                "provisional_amount": f"\${financial.get('adjusted_payout', 0):,.2f} (subject to adjustment)"
            }
        else:  # DENY
            return {
                "authorization_status": "✗ DENIED",
                "authorized_amount": "\$0.00",
                "denial_reason": decision.get("reasoning"),
                "policy_citations": decision.get("policy_citations", []),
                "appeal_rights": "You have the right to appeal this decision within 60 days",
                "appeal_contact": "appeals@insurance.com or 1-800-APPEAL-1"
            }
    
    def _generate_required_docs(self, decision: Dict) -> List[str]:
        """Generate list of required documentation based on decision"""
        base_docs = [
            "✓ Completed claim form with original signature",
            "✓ Copy of driver's license or government-issued ID",
            "✓ Vehicle registration and insurance card",
            "✓ Photographs of all damaged areas (minimum 6 angles)",
            "✓ Police report or incident report (if applicable)"
        ]
        
        if decision.get("outcome") == "APPROVE":
            base_docs.extend([
                "✓ Final itemized repair invoice on shop letterhead",
                "✓ Proof of deductible payment receipt",
                "✓ Before and after repair photographs",
                "✓ Parts receipts and warranty information",
                "✓ Odometer statement at time of repair"
            ])
        elif decision.get("outcome") == "INVESTIGATE":
            base_docs.extend([
                "✓ Comprehensive professional inspection report",
                "✓ Frame/structural integrity certification",
                "✓ Detailed photographs of suspected structural damage",
                "✓ Original repair estimate from certified facility",
                "✓ Vehicle history report (CARFAX or AutoCheck)"
            ])
        
        return base_docs
    
    def _generate_next_steps(self, decision: Dict) -> List[str]:
        """Generate next steps based on decision outcome"""
        if decision.get("outcome") == "APPROVE":
            return [
                "1. Select an approved repair facility or use your preferred shop",
                "2. Submit required documentation via online portal",
                "3. Pay your deductible directly to the repair facility",
                "4. Authorize repairs to begin",
                "5. Payment will be issued upon verification of completed repairs"
            ]
        elif decision.get("outcome") == "INVESTIGATE":
            return [
                "1. Wait for claims adjuster to contact you within 24 hours",
                "2. Schedule vehicle inspection at mutually convenient time",
                "3. Provide access to vehicle for comprehensive assessment",
                "4. Submit any additional documentation requested",
                "5. Decision will be updated within 3-5 business days after inspection"
            ]
        else:  # DENY
            return [
                "1. Review the denial reason and policy citations provided",
                "2. Gather any additional evidence to support your claim",
                "3. Contact our appeals department if you wish to appeal",
                "4. Submit appeal with supporting documentation within 60 days",
                "5. Consider alternative coverage options if applicable"
            ]



agent_instance = AutoInsuranceClaimsAgent()


agent_instructions="""
        You are an expert Auto Insurance Claims Settlement Agent powered by AWS Strands.
        
        Your responsibilities:
        1. Verify coverage limits and policy compliance
        2. Calculate accurate deductible adjustments
        3. Make fair and policy-compliant decisions (APPROVE/DENY/INVESTIGATE)
        4. Generate comprehensive audit documentation
        5. Produce clear, actionable settlement summaries
        
        Decision Guidelines:
        - APPROVE claims under \$5K with no structural damage
        - INVESTIGATE claims with frame damage, structural issues, or amounts >\$5K
        - DENY claims exceeding coverage limits with proper policy citations
        
        Always prioritize accuracy, fairness, regulatory compliance, and customer clarity.
        """
# settlement-agent.py
settlement_agent = Agent(
        name="AutoInsuranceClaimsSettlementAgent",
        description="Intelligent agent for automated auto insurance claims processing and settlement decisions",
        model=model,
        system_prompt=agent_instructions,
        tools=[
            agent_instance.verify_coverage_limits,
            agent_instance.apply_deductible_adjustments,
            agent_instance.evaluate_claim_decision,
            agent_instance.create_claim_documentation,
            agent_instance.generate_settlement_output
        ]
    )

################# A2A ################
app = FastAPI()
runtime_url = os.environ.get('AGENTCORE_RUNTIME_URL', 'http://127.0.0.1:9000/')
host, port = "0.0.0.0", 9000

a2a_server = A2AServer(
    agent=settlement_agent,
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


