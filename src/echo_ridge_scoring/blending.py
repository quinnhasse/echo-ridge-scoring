"""
Score blending system for combining AI and deterministic scores.

Provides configurable strategies for hybrid scoring with divergence detection.
"""

from typing import Dict, List, Any, Optional
import math


def blend_scores(
    ai_score: Dict[str, Any], 
    deterministic_score: Dict[str, Any], 
    strategy: str = "weighted_average",
    ai_weight: float = 0.5,
    divergence_threshold: float = 0.3,
    **config
) -> Dict[str, Any]:
    """
    Blend AI and deterministic scores using configurable strategies.
    
    Args:
        ai_score: AI scoring result (Roman's format)
        deterministic_score: Echo Ridge deterministic scoring result
        strategy: Blending strategy ('weighted_average', 'max_confidence', 'consensus')
        ai_weight: Weight for AI score in weighted_average (0.0-1.0)
        divergence_threshold: Threshold for flagging significant divergence (0.0-1.0)
        **config: Additional strategy-specific configuration
        
    Returns:
        Dict containing blended results with metadata
    """
    if not (0.0 <= ai_weight <= 1.0):
        raise ValueError("ai_weight must be between 0.0 and 1.0")
    if not (0.0 <= divergence_threshold <= 1.0):
        raise ValueError("divergence_threshold must be between 0.0 and 1.0")
    
    # Normalize scores to common scale (0.0-1.0)
    ai_normalized = _normalize_ai_score(ai_score)
    det_normalized = _normalize_deterministic_score(deterministic_score)
    
    # Calculate divergence
    divergence_info = _calculate_divergence(ai_normalized, det_normalized, divergence_threshold)
    
    # Apply blending strategy
    if strategy == "weighted_average":
        blended = _blend_weighted_average(ai_normalized, det_normalized, ai_weight)
    elif strategy == "max_confidence":
        blended = _blend_max_confidence(ai_normalized, det_normalized)
    elif strategy == "consensus":
        blended = _blend_consensus(ai_normalized, det_normalized, **config)
    else:
        raise ValueError(f"Unknown blending strategy: {strategy}")
    
    # Build result
    result = {
        "strategy": strategy,
        "config": {
            "ai_weight": ai_weight,
            "divergence_threshold": divergence_threshold,
            **config
        },
        "blended_score": blended,
        "component_scores": {
            "ai_score": ai_normalized,
            "deterministic_score": det_normalized
        },
        "divergence": divergence_info,
        "metadata": {
            "ai_source": ai_score.get("model", "unknown"),
            "deterministic_version": deterministic_score.get("metadata", {}).get("version", {}).get("engine", "unknown"),
            "company_id": deterministic_score.get("company_id", "unknown")
        }
    }
    
    return result


def _normalize_ai_score(ai_score: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize AI score to standard format (0.0-1.0 scale)."""
    normalized = {
        "overall_score": 0.0,
        "subscores": {},
        "confidence": 0.8,  # Default AI confidence
        "explanation": ""
    }
    
    # Extract overall score
    if "overall_score" in ai_score:
        # Already 0.0-1.0 scale
        normalized["overall_score"] = float(ai_score["overall_score"])
    elif "score_summary" in ai_score and "weighted_overall" in ai_score["score_summary"]:
        normalized["overall_score"] = float(ai_score["score_summary"]["weighted_overall"])
    
    # Extract component scores (DIMB)
    if "dimb_scores" in ai_score:
        for component, score_info in ai_score["dimb_scores"].items():
            if isinstance(score_info, dict) and "value" in score_info:
                normalized["subscores"][component] = {
                    "score": float(score_info["value"]),
                    "confidence": score_info.get("confidence", 0.8),
                    "evidence": score_info.get("evidence", "")
                }
    
    # Extract explanation
    if "overall_note" in ai_score:
        normalized["explanation"] = str(ai_score["overall_note"])
    
    return normalized


def _normalize_deterministic_score(det_score: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize deterministic score to standard format (0.0-1.0 scale)."""
    normalized = {
        "overall_score": 0.0,
        "subscores": {},
        "confidence": 1.0,  # Deterministic = full confidence
        "explanation": ""
    }
    
    # Convert 0-100 scale to 0.0-1.0
    if "final_score" in det_score:
        normalized["overall_score"] = float(det_score["final_score"]) / 100.0
    
    # Extract component scores
    if "subscores" in det_score:
        component_mapping = {
            "digital": "D",
            "ops": "O", 
            "info_flow": "I",
            "market": "M",
            "budget": "B"
        }
        
        for det_component, ai_component in component_mapping.items():
            if det_component in det_score["subscores"]:
                subscore_info = det_score["subscores"][det_component]
                if isinstance(subscore_info, dict) and "score" in subscore_info:
                    # Convert to 0.0-1.0 scale
                    score_value = float(subscore_info["score"]) / 100.0
                    normalized["subscores"][ai_component] = {
                        "score": score_value,
                        "confidence": subscore_info.get("confidence", 1.0),
                        "evidence": f"Deterministic calculation: {score_value:.2f}"
                    }
    
    # Extract explanation
    if "explanation" in det_score:
        normalized["explanation"] = str(det_score["explanation"])
    
    return normalized


def _calculate_divergence(ai_score: Dict[str, Any], det_score: Dict[str, Any], 
                        threshold: float) -> Dict[str, Any]:
    """Calculate divergence between AI and deterministic scores."""
    divergences = {}
    flags = []
    
    # Overall score divergence
    ai_overall = ai_score.get("overall_score", 0.0)
    det_overall = det_score.get("overall_score", 0.0)
    overall_divergence = abs(ai_overall - det_overall)
    
    divergences["overall"] = overall_divergence
    if overall_divergence > threshold:
        flags.append(f"Overall score divergence: {overall_divergence:.3f} > {threshold}")
    
    # Component score divergences
    ai_subscores = ai_score.get("subscores", {})
    det_subscores = det_score.get("subscores", {})
    
    for component in ["D", "O", "I", "M", "B"]:
        ai_sub = ai_subscores.get(component, {}).get("score", 0.0)
        det_sub = det_subscores.get(component, {}).get("score", 0.0)
        
        component_divergence = abs(ai_sub - det_sub)
        divergences[f"component_{component}"] = component_divergence
        
        if component_divergence > threshold:
            flags.append(f"Component {component} divergence: {component_divergence:.3f} > {threshold}")
    
    return {
        "divergences": divergences,
        "flags": flags,
        "has_significant_divergence": len(flags) > 0,
        "max_divergence": max(divergences.values()) if divergences else 0.0
    }


def _blend_weighted_average(ai_score: Dict[str, Any], det_score: Dict[str, Any], 
                          ai_weight: float) -> Dict[str, Any]:
    """Blend scores using weighted average."""
    det_weight = 1.0 - ai_weight
    
    # Blend overall score
    ai_overall = ai_score.get("overall_score", 0.0)
    det_overall = det_score.get("overall_score", 0.0)
    blended_overall = (ai_overall * ai_weight) + (det_overall * det_weight)
    
    # Blend component scores
    blended_subscores = {}
    ai_subscores = ai_score.get("subscores", {})
    det_subscores = det_score.get("subscores", {})
    
    for component in ["D", "O", "I", "M", "B"]:
        ai_sub = ai_subscores.get(component, {}).get("score", 0.0)
        det_sub = det_subscores.get(component, {}).get("score", 0.0)
        
        blended_sub = (ai_sub * ai_weight) + (det_sub * det_weight)
        
        # Combine confidences
        ai_conf = ai_subscores.get(component, {}).get("confidence", 0.8)
        det_conf = det_subscores.get(component, {}).get("confidence", 1.0)
        blended_conf = (ai_conf * ai_weight) + (det_conf * det_weight)
        
        # Combine evidence
        ai_evidence = ai_subscores.get(component, {}).get("evidence", "")
        det_evidence = det_subscores.get(component, {}).get("evidence", "")
        combined_evidence = f"AI: {ai_evidence} | Det: {det_evidence}"
        
        blended_subscores[component] = {
            "score": blended_sub,
            "confidence": blended_conf,
            "evidence": combined_evidence,
            "weights_used": {"ai": ai_weight, "deterministic": det_weight}
        }
    
    # Blend explanations
    ai_explanation = ai_score.get("explanation", "")
    det_explanation = det_score.get("explanation", "")
    blended_explanation = f"Weighted blend (AI: {ai_weight:.1f}, Det: {det_weight:.1f}). AI: {ai_explanation} | Deterministic: {det_explanation}"
    
    return {
        "overall_score": blended_overall,
        "subscores": blended_subscores,
        "confidence": (ai_score.get("confidence", 0.8) * ai_weight) + (det_score.get("confidence", 1.0) * det_weight),
        "explanation": blended_explanation
    }


def _blend_max_confidence(ai_score: Dict[str, Any], det_score: Dict[str, Any]) -> Dict[str, Any]:
    """Blend scores by selecting highest confidence score for each component."""
    ai_conf = ai_score.get("confidence", 0.8)
    det_conf = det_score.get("confidence", 1.0)
    
    # Select overall score based on higher confidence
    if ai_conf > det_conf:
        overall_score = ai_score.get("overall_score", 0.0)
        overall_source = "ai"
    else:
        overall_score = det_score.get("overall_score", 0.0)  
        overall_source = "deterministic"
    
    # Select component scores based on individual confidences
    blended_subscores = {}
    ai_subscores = ai_score.get("subscores", {})
    det_subscores = det_score.get("subscores", {})
    
    for component in ["D", "O", "I", "M", "B"]:
        ai_comp_conf = ai_subscores.get(component, {}).get("confidence", 0.8)
        det_comp_conf = det_subscores.get(component, {}).get("confidence", 1.0)
        
        if ai_comp_conf > det_comp_conf:
            selected = ai_subscores.get(component, {})
            source = "ai"
        else:
            selected = det_subscores.get(component, {})
            source = "deterministic"
        
        blended_subscores[component] = {
            "score": selected.get("score", 0.0),
            "confidence": selected.get("confidence", 0.8),
            "evidence": selected.get("evidence", ""),
            "selected_source": source
        }
    
    return {
        "overall_score": overall_score,
        "subscores": blended_subscores,
        "confidence": max(ai_conf, det_conf),
        "explanation": f"Max confidence selection. Overall from: {overall_source}",
        "selection_metadata": {
            "overall_source": overall_source,
            "ai_confidence": ai_conf,
            "deterministic_confidence": det_conf
        }
    }


def _blend_consensus(ai_score: Dict[str, Any], det_score: Dict[str, Any], 
                   consensus_threshold: float = 0.2, **config) -> Dict[str, Any]:
    """Blend scores using consensus approach - average when close, flag when divergent."""
    # Calculate divergences for consensus decision
    ai_overall = ai_score.get("overall_score", 0.0)
    det_overall = det_score.get("overall_score", 0.0)
    overall_divergence = abs(ai_overall - det_overall)
    
    # Overall consensus
    if overall_divergence <= consensus_threshold:
        blended_overall = (ai_overall + det_overall) / 2.0
        overall_consensus = True
    else:
        # No consensus - use deterministic as more reliable
        blended_overall = det_overall
        overall_consensus = False
    
    # Component consensus
    blended_subscores = {}
    ai_subscores = ai_score.get("subscores", {})
    det_subscores = det_score.get("subscores", {})
    
    for component in ["D", "O", "I", "M", "B"]:
        ai_sub = ai_subscores.get(component, {}).get("score", 0.0)
        det_sub = det_subscores.get(component, {}).get("score", 0.0)
        
        component_divergence = abs(ai_sub - det_sub)
        
        if component_divergence <= consensus_threshold:
            # Consensus - average the scores
            blended_sub = (ai_sub + det_sub) / 2.0
            consensus_reached = True
            evidence = f"Consensus reached: AI={ai_sub:.2f}, Det={det_sub:.2f}, Avg={blended_sub:.2f}"
        else:
            # No consensus - use deterministic
            blended_sub = det_sub
            consensus_reached = False
            evidence = f"No consensus (divergence={component_divergence:.2f}): Using deterministic={det_sub:.2f}"
        
        blended_subscores[component] = {
            "score": blended_sub,
            "confidence": 0.9 if consensus_reached else 0.7,
            "evidence": evidence,
            "consensus_reached": consensus_reached,
            "divergence": component_divergence
        }
    
    return {
        "overall_score": blended_overall,
        "subscores": blended_subscores,
        "confidence": 0.9 if overall_consensus else 0.7,
        "explanation": f"Consensus blending (threshold={consensus_threshold}). Overall consensus: {overall_consensus}",
        "consensus_metadata": {
            "overall_consensus": overall_consensus,
            "overall_divergence": overall_divergence,
            "consensus_threshold": consensus_threshold,
            "components_with_consensus": sum(1 for s in blended_subscores.values() if s.get("consensus_reached", False))
        }
    }