"""Map jailbreak vulnerabilities to regulatory frameworks."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from src.models.jailbreak import ViolationDomain, SeverityLevel, EvaluationResult
from src.utils.logger import log


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks."""
    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    CWE = "cwe"
    ISO_27001 = "iso_27001"


@dataclass
class ViolationMapping:
    """Mapping of a vulnerability to regulatory requirements."""
    framework: RegulatoryFramework
    requirement_id: str
    requirement_name: str
    description: str
    severity: SeverityLevel
    violation_domains: List[ViolationDomain]
    mitigation_guidance: Optional[str] = None


class ComplianceMapper:
    """
    Maps jailbreak vulnerabilities to regulatory frameworks.
    
    Supports:
    - EU AI Act compliance
    - NIST AI Risk Management Framework
    - CWE (Common Weakness Enumeration)
    - ISO 27001
    """
    
    def __init__(self):
        """Initialize compliance mapper with framework mappings."""
        self.eu_ai_act_mappings = self._load_eu_ai_act_mappings()
        self.nist_rmf_mappings = self._load_nist_rmf_mappings()
        self.cwe_mappings = self._load_cwe_mappings()
    
    def map_violation(
        self,
        evaluation_result: EvaluationResult
    ) -> List[ViolationMapping]:
        """
        Map an evaluation result to regulatory violations.
        
        Args:
            evaluation_result: Evaluation result to map
            
        Returns:
            List of violation mappings
        """
        if not evaluation_result.is_jailbroken:
            return []
        
        mappings = []
        
        # Map to EU AI Act
        eu_mappings = self._map_to_eu_ai_act(evaluation_result)
        mappings.extend(eu_mappings)
        
        # Map to NIST RMF
        nist_mappings = self._map_to_nist_rmf(evaluation_result)
        mappings.extend(nist_mappings)
        
        # Map to CWE
        cwe_mappings = self._map_to_cwe(evaluation_result)
        mappings.extend(cwe_mappings)
        
        return mappings
    
    def _map_to_eu_ai_act(
        self,
        result: EvaluationResult
    ) -> List[ViolationMapping]:
        """Map violation to EU AI Act requirements."""
        mappings = []
        
        # Article 5: Prohibited AI Practices
        if ViolationDomain.BIOLOGICAL in result.violation_domains:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.EU_AI_ACT,
                requirement_id="Art.5.1.b",
                requirement_name="Prohibited: AI systems that exploit vulnerabilities",
                description="AI system exploited to generate biological weapon information",
                severity=result.severity,
                violation_domains=[ViolationDomain.BIOLOGICAL],
                mitigation_guidance="Implement strict content filtering for biological/chemical topics"
            ))
        
        if ViolationDomain.FRAUD in result.violation_domains:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.EU_AI_ACT,
                requirement_id="Art.5.1.d",
                requirement_name="Prohibited: AI systems for social scoring",
                description="AI system exploited for fraudulent activities",
                severity=result.severity,
                violation_domains=[ViolationDomain.FRAUD],
                mitigation_guidance="Add fraud detection and prevention mechanisms"
            ))
        
        # Article 10: Data Governance
        if ViolationDomain.PRIVACY in result.violation_domains:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.EU_AI_ACT,
                requirement_id="Art.10",
                requirement_name="Data Governance Requirements",
                description="AI system exploited to leak private information",
                severity=result.severity,
                violation_domains=[ViolationDomain.PRIVACY],
                mitigation_guidance="Implement data privacy controls and access restrictions"
            ))
        
        # Article 15: Human Oversight
        if result.severity.value >= SeverityLevel.HIGH.value:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.EU_AI_ACT,
                requirement_id="Art.15",
                requirement_name="Human Oversight Requirements",
                description="High-severity jailbreak indicates insufficient human oversight",
                severity=result.severity,
                violation_domains=result.violation_domains,
                mitigation_guidance="Implement human-in-the-loop oversight for high-risk outputs"
            ))
        
        return mappings
    
    def _map_to_nist_rmf(
        self,
        result: EvaluationResult
    ) -> List[ViolationMapping]:
        """Map violation to NIST AI Risk Management Framework."""
        mappings = []
        
        # MAP Function: Measure, Analyze, and Prioritize
        mappings.append(ViolationMapping(
            framework=RegulatoryFramework.NIST_AI_RMF,
            requirement_id="MAP-1",
            requirement_name="MAP-1: Context Establishment",
            description="Jailbreak indicates insufficient context understanding",
            severity=result.severity,
            violation_domains=result.violation_domains,
            mitigation_guidance="Establish comprehensive context for AI system deployment"
        ))
        
        # MEASURE Function
        if result.severity.value >= SeverityLevel.MODERATE.value:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.NIST_AI_RMF,
                requirement_id="MEASURE-2",
                requirement_name="MEASURE-2: Performance and Metrics",
                description="Jailbreak indicates inadequate performance metrics",
                severity=result.severity,
                violation_domains=result.violation_domains,
                mitigation_guidance="Implement comprehensive safety and security metrics"
            ))
        
        # MANAGE Function
        mappings.append(ViolationMapping(
            framework=RegulatoryFramework.NIST_AI_RMF,
            requirement_id="MANAGE-3",
            requirement_name="MANAGE-3: Risk Management",
            description="Jailbreak represents unmanaged risk",
            severity=result.severity,
            violation_domains=result.violation_domains,
            mitigation_guidance="Establish continuous risk monitoring and management processes"
        ))
        
        return mappings
    
    def _map_to_cwe(
        self,
        result: EvaluationResult
    ) -> List[ViolationMapping]:
        """Map violation to CWE (Common Weakness Enumeration)."""
        mappings = []
        
        # CWE-79: Improper Neutralization of Input
        mappings.append(ViolationMapping(
            framework=RegulatoryFramework.CWE,
            requirement_id="CWE-79",
            requirement_name="Improper Neutralization of Input During Web Page Generation",
            description="Jailbreak prompt injection indicates improper input sanitization",
            severity=result.severity,
            violation_domains=result.violation_domains,
            mitigation_guidance="Implement proper input validation and sanitization"
        ))
        
        # CWE-20: Improper Input Validation
        if "inversion" in result.attack_strategy.value or "probing" in result.attack_strategy.value:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.CWE,
                requirement_id="CWE-20",
                requirement_name="Improper Input Validation",
                description="Attack strategy exploits lack of input validation",
                severity=result.severity,
                violation_domains=result.violation_domains,
                mitigation_guidance="Implement comprehensive input validation and filtering"
            ))
        
        # CWE-434: Unrestricted Upload of File with Dangerous Type
        if ViolationDomain.CYBER in result.violation_domains:
            mappings.append(ViolationMapping(
                framework=RegulatoryFramework.CWE,
                requirement_id="CWE-434",
                requirement_name="Unrestricted Upload of File with Dangerous Type",
                description="Cyber-related jailbreak may enable dangerous content generation",
                severity=result.severity,
                violation_domains=[ViolationDomain.CYBER],
                mitigation_guidance="Restrict generation of potentially dangerous content"
            ))
        
        return mappings
    
    def generate_compliance_report(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Compliance report dictionary
        """
        successful_exploits = [r for r in evaluation_results if r.is_jailbroken]
        
        # Map all violations
        all_mappings = []
        for result in successful_exploits:
            mappings = self.map_violation(result)
            all_mappings.extend(mappings)
        
        # Group by framework
        by_framework = {}
        for mapping in all_mappings:
            framework = mapping.framework.value
            if framework not in by_framework:
                by_framework[framework] = []
            by_framework[framework].append(mapping)
        
        # Count violations by severity
        severity_counts = {}
        for mapping in all_mappings:
            severity = mapping.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report = {
            "total_violations": len(all_mappings),
            "total_exploits": len(successful_exploits),
            "violations_by_framework": {
                framework: len(mappings)
                for framework, mappings in by_framework.items()
            },
            "violations_by_severity": severity_counts,
            "critical_violations": [
                {
                    "framework": m.framework.value,
                    "requirement_id": m.requirement_id,
                    "requirement_name": m.requirement_name,
                    "severity": m.severity.value,
                    "mitigation": m.mitigation_guidance
                }
                for m in all_mappings
                if m.severity.value >= SeverityLevel.HIGH.value
            ],
            "detailed_mappings": [
                {
                    "framework": m.framework.value,
                    "requirement_id": m.requirement_id,
                    "requirement_name": m.requirement_name,
                    "description": m.description,
                    "severity": m.severity.value,
                    "violation_domains": [d.value for d in m.violation_domains],
                    "mitigation_guidance": m.mitigation_guidance
                }
                for m in all_mappings
            ]
        }
        
        log.info(
            f"Generated compliance report: {len(all_mappings)} violations "
            f"across {len(by_framework)} frameworks"
        )
        
        return report
    
    def _load_eu_ai_act_mappings(self) -> Dict[str, Any]:
        """Load EU AI Act mapping rules."""
        # In production, this would load from a JSON file
        return {}
    
    def _load_nist_rmf_mappings(self) -> Dict[str, Any]:
        """Load NIST RMF mapping rules."""
        # In production, this would load from a JSON file
        return {}
    
    def _load_cwe_mappings(self) -> Dict[str, Any]:
        """Load CWE mapping rules."""
        # In production, this would load from a JSON file
        return {}

