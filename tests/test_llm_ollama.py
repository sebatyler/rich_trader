import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic import Field

from core.llm import invoke_llm_ollama
from core.llm import ollama_client


class _MockRecommendation(BaseModel):
    scratchpad: str = Field(..., description="Analysis")
    reasoning: str = Field(..., description="Reasoning")
    recommendations: list = Field(default_factory=list, description="Recommendations")


@pytest.fixture
def mock_ollama_response():
    return MagicMock(
        message=MagicMock(
            content="""
scratchpad: |
  Test analysis
reasoning: |
  Test reasoning
recommendations: []
"""
        )
    )


def test_invoke_llm_ollama_parses_yaml_into_model(mock_ollama_response):
    with patch.object(ollama_client, "chat", return_value=mock_ollama_response) as mock_chat:
        result = invoke_llm_ollama(
            "You are a trading advisor",
            "Market data: {test_data}",
            model=_MockRecommendation,
            template_format="f-string",
            test_data="BTC is up 5%",
        )

    assert isinstance(result, _MockRecommendation)
    assert result.scratchpad.strip() == "Test analysis"
    assert result.reasoning.strip() == "Test reasoning"
    assert result.recommendations == []

    call_args = mock_chat.call_args[1]
    assert call_args["model"] == os.getenv("OLLAMA_MODEL", "deepseek-v4-pro")
    assert call_args["stream"] is False
    messages = call_args["messages"]
    assert messages[0]["role"] == "system"
    assert "trading advisor" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "BTC is up 5%" in messages[1]["content"]


def test_invoke_llm_ollama_returns_raw_string_when_no_model(mock_ollama_response):
    with patch.object(ollama_client, "chat", return_value=mock_ollama_response) as mock_chat:
        result = invoke_llm_ollama(
            "System prompt",
            "User prompt",
        )

    assert isinstance(result, str)
    assert "scratchpad" in result
    mock_chat.assert_called_once()


@pytest.mark.skipif(
    not os.getenv("OLLAMA_API_KEY"),
    reason="No OLLAMA_API_KEY set",
)
def test_invoke_llm_ollama_real_api():
    result = invoke_llm_ollama(
        "You are a helpful assistant. Reply with exactly one word.",
        "Say hello.",
    )
    assert isinstance(result, str)
    assert "hello" in result.lower()
