#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates aligned with the original scoring prompts."""

from __future__ import annotations

from typing import Dict


class PromptTemplateManager:
    DIRECT_SCORING_WITH_REPO_TEMPLATE = """You are a professional code understanding evaluation expert. Please objectively and fairly score the model's answer based on the following information.

## Question Information
**Question Description**: {description}

## Real Repository Information
{repo_info_section}

## Model Answer to be Scored
**Model**: {model_name}
**Answer**: {model_answer}

## Reference Answer
{reference_text}

## Scoring Criteria
{scoring_criteria}

## Scoring Requirements
1. Carefully read the question description, reference answer, repository information, and scoring criteria
2. **First check if the model's answer is related to the repository information; if completely unrelated, give 0 points directly**
3. **CRITICAL: Only use integer scores that exist in the scoring criteria. DO NOT create non-existent score ranges! DO NOT use decimal scores like 0.5, 1.5, etc.!**
4. Max knowledge score is 9 by default, and max clarity score is 1 by default
5. Score each model's answer independently, but maintain consistency in scoring standards (Score strictly according to the criteria, avoid inflated scores!!! Try not to give full marks unless it fits the criteria very well!)
6. Focus on accuracy, completeness, logic, conciseness, and human-like quality of model answers
7. **Line numbers in answers (e.g., @file.py:10-20) are NORMAL and NOT hallucinations. Only fabricated non-existent files or paths are hallucinations.** (Model output may include absolute path prefix /app/repo/, consider this prefix when evaluating paths)
8. **Check if the model outputs non-existent file paths (hallucination), deduct at most 1 point per scoring dimension for hallucinations**
9. Provide specific scores and detailed reasons based on the scoring criteria
10. **Must output scoring results in standard JSON format with reason BEFORE scores**

## Scoring Output Format
```json
{{
    "model_name": "{model_name}",
    "reason": "<detailed scoring reason - EXPLAIN FIRST before giving scores>",
    "knowledge_score": <integer only, from scoring criteria>,
    "knowledge_max": <integer only, from scoring criteria>,
    "clarity_score": <integer only, from scoring criteria>,
    "clarity_max": <integer only, from scoring criteria>,
    "total_score": <integer only>,
    "max_score": <integer only>,
    "hallucination": <true/false, only if fabricated files/paths exist>
}}
```
"""

    @staticmethod
    def _repo_info_section(repo_info: Dict[str, str] | None) -> str:
        repo_info = repo_info or {}
        directory_structure = repo_info.get("directory_structure", "").strip()
        return f"**Directory Structure**:\n```\n{directory_structure}\n```"

    @staticmethod
    def create_scoring_prompt(question, target_answer, repo_info: Dict[str, str] | None = None) -> str:
        return PromptTemplateManager.DIRECT_SCORING_WITH_REPO_TEMPLATE.format(
            description=question.description,
            repo_info_section=PromptTemplateManager._repo_info_section(repo_info),
            reference_text=question.reference_text,
            scoring_criteria=question.scoring_criteria,
            model_name=target_answer.model_name,
            model_answer=target_answer.answer,
        )
