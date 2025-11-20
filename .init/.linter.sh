#!/bin/bash
cd /home/kavia/workspace/code-generation/career-navigator-platform-209461-209473/llm_recommendation_service
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

