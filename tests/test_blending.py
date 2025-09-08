"""
Tests for score blending system.
"""

import pytest
from src.echo_ridge_scoring.blending import blend_scores


# Sample scores for testing
SAMPLE_AI_SCORE = {
    "model": "gpt-4o",
    "overall_note": "Strong digital presence with good operational foundation",
    "dimb_scores": {
        "D": {"value": 0.7, "evidence": "Modern website with portal", "confidence": 0.8},
        "O": {"value": 0.6, "evidence": "Multiple locations", "confidence": 0.8}, 
        "I": {"value": 0.4, "evidence": "Basic systems", "confidence": 0.7},
        "M": {"value": 0.5, "evidence": "Competitive market", "confidence": 0.8},
        "B": {"value": 0.6, "evidence": "Premium pricing", "confidence": 0.8}
    },
    "overall_score": 0.58,
    "score_summary": {"weighted_overall": 0.58}
}

SAMPLE_DETERMINISTIC_SCORE = {
    "final_score": 72.5,  # 0.725 on 0-1 scale
    "subscores": {
        "digital": {"score": 80.0, "confidence": 0.9},
        "ops": {"score": 65.0, "confidence": 1.0},
        "info_flow": {"score": 70.0, "confidence": 1.0},
        "market": {"score": 75.0, "confidence": 1.0},  
        "budget": {"score": 68.0, "confidence": 0.8}
    },
    "explanation": "Deterministic scoring based on quantitative metrics",
    "company_id": "test-company-001",
    "metadata": {
        "version": {"engine": "1.1.0", "api": "1.1.0"},
        "processing_time_ms": 45.2
    }
}


class TestBlending:
    """Test score blending strategies."""
    
    def test_weighted_average_default(self):
        """Test weighted average blending with default 50/50 weights."""
        result = blend_scores(SAMPLE_AI_SCORE, SAMPLE_DETERMINISTIC_SCORE)
        
        # Verify strategy and config
        assert result["strategy"] == "weighted_average"
        assert result["config"]["ai_weight"] == 0.5
        
        # Check blended overall score (0.58 * 0.5 + 0.725 * 0.5 = 0.6525)
        expected_overall = (0.58 * 0.5) + (0.725 * 0.5)
        assert abs(result["blended_score"]["overall_score"] - expected_overall) < 0.001
        
        # Verify component scores are blended
        blended_d = result["blended_score"]["subscores"]["D"]
        expected_d = (0.7 * 0.5) + (0.8 * 0.5)  # AI: 0.7, Det: 0.8 (80/100)
        assert abs(blended_d["score"] - expected_d) < 0.001
        
    def test_weighted_average_custom_weights(self):
        """Test weighted average with custom AI weight."""
        result = blend_scores(
            SAMPLE_AI_SCORE, 
            SAMPLE_DETERMINISTIC_SCORE,
            ai_weight=0.3
        )
        
        # Check custom weight applied
        assert result["config"]["ai_weight"] == 0.3
        
        # Check calculation (0.58 * 0.3 + 0.725 * 0.7 = 0.6815)
        expected_overall = (0.58 * 0.3) + (0.725 * 0.7)
        assert abs(result["blended_score"]["overall_score"] - expected_overall) < 0.001
        
    def test_max_confidence_strategy(self):
        """Test max confidence blending strategy."""
        result = blend_scores(
            SAMPLE_AI_SCORE,
            SAMPLE_DETERMINISTIC_SCORE, 
            strategy="max_confidence"
        )
        
        assert result["strategy"] == "max_confidence"
        
        # Deterministic should win overall (confidence 1.0 vs 0.8)
        assert result["blended_score"]["overall_score"] == 0.725  # Deterministic
        
        # Check component selection based on confidence
        blended_d = result["blended_score"]["subscores"]["D"]
        # Det confidence (1.0) > AI confidence (0.8), so should use Det (0.8)
        assert blended_d["score"] == 0.8  # 80/100
        assert blended_d["selected_source"] == "deterministic"
        
    def test_consensus_strategy_agreement(self):
        """Test consensus strategy when scores agree."""
        # Create closer scores for consensus
        close_ai = SAMPLE_AI_SCORE.copy()
        close_ai["dimb_scores"]["D"]["value"] = 0.75  # Close to det 0.8
        
        result = blend_scores(
            close_ai,
            SAMPLE_DETERMINISTIC_SCORE,
            strategy="consensus",
            consensus_threshold=0.1
        )
        
        assert result["strategy"] == "consensus"
        
        # D component should reach consensus (|0.75 - 0.8| = 0.05 < 0.1)
        blended_d = result["blended_score"]["subscores"]["D"]
        assert blended_d["consensus_reached"] == True
        expected_d = (0.75 + 0.8) / 2  # Average
        assert abs(blended_d["score"] - expected_d) < 0.001
        
    def test_consensus_strategy_disagreement(self):
        """Test consensus strategy when scores diverge."""
        result = blend_scores(
            SAMPLE_AI_SCORE,
            SAMPLE_DETERMINISTIC_SCORE, 
            strategy="consensus",
            consensus_threshold=0.05  # Strict threshold
        )
        
        # D component divergence: |0.7 - 0.8| = 0.1 > 0.05
        blended_d = result["blended_score"]["subscores"]["D"]
        assert blended_d["consensus_reached"] == False
        assert blended_d["score"] == 0.8  # Uses deterministic
        
    def test_divergence_detection(self):
        """Test divergence detection and flagging.""" 
        # Create highly divergent scores
        divergent_ai = SAMPLE_AI_SCORE.copy()
        divergent_ai["dimb_scores"]["D"]["value"] = 0.2  # Very different from det 0.8
        
        result = blend_scores(
            divergent_ai,
            SAMPLE_DETERMINISTIC_SCORE,
            divergence_threshold=0.3
        )
        
        # Check divergence detection
        divergence = result["divergence"]
        assert divergence["has_significant_divergence"] == True
        assert len(divergence["flags"]) > 0
        
        # D component should be flagged (|0.2 - 0.8| = 0.6 > 0.3)
        assert divergence["divergences"]["component_D"] == 0.6
        assert any("Component D divergence" in flag for flag in divergence["flags"])
        
    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown blending strategy"):
            blend_scores(
                SAMPLE_AI_SCORE,
                SAMPLE_DETERMINISTIC_SCORE,
                strategy="invalid_strategy"
            )
            
    def test_invalid_ai_weight(self):
        """Test validation of ai_weight parameter."""
        with pytest.raises(ValueError, match="ai_weight must be between 0.0 and 1.0"):
            blend_scores(
                SAMPLE_AI_SCORE,
                SAMPLE_DETERMINISTIC_SCORE,
                ai_weight=1.5
            )
            
    def test_invalid_divergence_threshold(self):
        """Test validation of divergence_threshold parameter."""
        with pytest.raises(ValueError, match="divergence_threshold must be between 0.0 and 1.0"):
            blend_scores(
                SAMPLE_AI_SCORE,
                SAMPLE_DETERMINISTIC_SCORE, 
                divergence_threshold=-0.1
            )
            
    def test_missing_ai_fields(self):
        """Test handling of missing AI score fields."""
        incomplete_ai = {"model": "gpt-4o"}  # Missing most fields
        
        result = blend_scores(incomplete_ai, SAMPLE_DETERMINISTIC_SCORE)
        
        # Should handle gracefully with defaults
        assert result["blended_score"]["overall_score"] > 0
        assert "D" in result["blended_score"]["subscores"]
        
    def test_missing_deterministic_fields(self):
        """Test handling of missing deterministic score fields."""
        incomplete_det = {"company_id": "test"}  # Missing most fields
        
        result = blend_scores(SAMPLE_AI_SCORE, incomplete_det)
        
        # Should handle gracefully
        assert result["blended_score"]["overall_score"] >= 0
        
    def test_score_normalization(self):
        """Test that scores are properly normalized to 0-1 scale."""
        result = blend_scores(SAMPLE_AI_SCORE, SAMPLE_DETERMINISTIC_SCORE)
        
        # All normalized scores should be 0-1
        ai_norm = result["component_scores"]["ai_score"]
        det_norm = result["component_scores"]["deterministic_score"]
        
        assert 0 <= ai_norm["overall_score"] <= 1
        assert 0 <= det_norm["overall_score"] <= 1
        
        for component in ["D", "O", "I", "M", "B"]:
            if component in ai_norm["subscores"]:
                assert 0 <= ai_norm["subscores"][component]["score"] <= 1
            if component in det_norm["subscores"]:
                assert 0 <= det_norm["subscores"][component]["score"] <= 1
                
    def test_metadata_preservation(self):
        """Test that metadata is preserved in blend result."""
        result = blend_scores(SAMPLE_AI_SCORE, SAMPLE_DETERMINISTIC_SCORE)
        
        metadata = result["metadata"]
        assert metadata["ai_source"] == "gpt-4o"
        assert metadata["deterministic_version"] == "1.1.0"
        assert metadata["company_id"] == "test-company-001"
        
    def test_confidence_blending(self):
        """Test that confidences are properly blended."""
        result = blend_scores(SAMPLE_AI_SCORE, SAMPLE_DETERMINISTIC_SCORE)
        
        # Weighted average should blend confidences
        blended_confidence = result["blended_score"]["confidence"]
        expected = (0.8 * 0.5) + (1.0 * 0.5)  # AI avg conf ~0.8, Det conf 1.0
        assert abs(blended_confidence - expected) < 0.1  # Allow small variance
        
    def test_evidence_combination(self):
        """Test that evidence strings are properly combined."""
        result = blend_scores(SAMPLE_AI_SCORE, SAMPLE_DETERMINISTIC_SCORE)
        
        # Check that evidence includes both AI and deterministic
        d_evidence = result["blended_score"]["subscores"]["D"]["evidence"]
        assert "AI:" in d_evidence
        assert "Det:" in d_evidence or "Deterministic" in d_evidence
        
    def test_edge_case_zero_scores(self):
        """Test handling of zero scores."""
        zero_ai = {
            "dimb_scores": {
                "D": {"value": 0.0, "confidence": 0.5}
            },
            "overall_score": 0.0
        }
        
        zero_det = {
            "final_score": 0.0,
            "subscores": {
                "digital": {"score": 0.0, "confidence": 1.0}
            }
        }
        
        result = blend_scores(zero_ai, zero_det)
        
        # Should handle zero scores without error
        assert result["blended_score"]["overall_score"] == 0.0
        assert result["blended_score"]["subscores"]["D"]["score"] == 0.0
        
    def test_edge_case_perfect_scores(self):
        """Test handling of perfect scores."""
        perfect_ai = {
            "dimb_scores": {
                "D": {"value": 1.0, "confidence": 1.0}
            },
            "overall_score": 1.0
        }
        
        perfect_det = {
            "final_score": 100.0,
            "subscores": {
                "digital": {"score": 100.0, "confidence": 1.0}
            }
        }
        
        result = blend_scores(perfect_ai, perfect_det)
        
        # Should handle perfect scores
        assert result["blended_score"]["overall_score"] == 1.0
        assert result["blended_score"]["subscores"]["D"]["score"] == 1.0