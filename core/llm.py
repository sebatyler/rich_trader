import json
import logging
import os
import re

from google import genai
from google.genai.types import GenerateContentConfig
from google.genai.types import GoogleSearch
from google.genai.types import Tool
from langchain.output_parsers import YamlOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from ollama import Client as OllamaClient

# Initialize the LLM
chat_anthropic = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-sonnet-20241022",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30,
    max_retries=0,
)

# https://ai.google.dev/gemini-api/docs/models
# https://ai.google.dev/gemini-api/docs/rate-limits
gemini_search_models = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
gemini_models = [] + gemini_search_models

chat_deepseek = ChatDeepSeek(
    # model="deepseek-chat",
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=90,
    max_retries=0,
)

chat_gemini_models = [
    ChatGoogleGenerativeAI(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=90,
        max_retries=0,
    )
    for model in gemini_models
]

llm_primary = chat_deepseek.with_fallbacks(chat_gemini_models)
llm_fallback = chat_gemini_models[1]

ollama_client = OllamaClient(
    host=os.getenv("OLLAMA_HOST", "https://ollama.com"),
    headers={"Authorization": "Bearer " + (os.getenv("OLLAMA_API_KEY") or "")},
)


def invoke_llm_ollama(
    prompt, *args, model=None, template_format="f-string", **kwargs
):
    """Invoke Ollama chat API and optionally parse YAML into a Pydantic model."""
    messages = [{"role": "system", "content": prompt}]
    for arg in args:
        if template_format == "f-string":
            content = arg.format(**kwargs)
        else:
            content = arg % kwargs
        messages.append({"role": "user", "content": content})

    ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-v4-pro")
    response = ollama_client.chat(
        model=ollama_model,
        messages=messages,
        stream=False,
    )
    raw = response.message.content
    logging.info(f"invoke_llm_ollama model={ollama_model}: {raw[:300]}...")

    if model:
        parser = YamlOutputParser(pydantic_object=model)
        return parser.parse(raw)
    return raw


def invoke_llm(
    prompt, *args, model=None, with_fallback=False, structured_output=False, template_format="f-string", **kwargs
):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            *[("human", arg) for arg in args],
        ],
        template_format=template_format,
    )
    llm = llm_fallback if with_fallback else llm_primary

    if model and structured_output:
        llm = llm.with_structured_output(model)

    # Combine the prompt with the structured LLM runnable
    chain = chat_prompt | llm

    if model and not structured_output:
        parser = YamlOutputParser(pydantic_object=model)
        chain = chain | parser

    # Invoke the runnable to get structured output
    result = chain.invoke(kwargs)
    logging.info(f"{with_fallback=}: {result=}")

    return result if model else result.content


def invoke_llm_thinking_mode(prompt, *args, **kwargs):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            *[("human", arg) for arg in args],
        ]
    )

    llm = chat_gemini_models[-1]

    chain = chat_prompt | llm

    # Invoke the runnable to get structured output
    result = chain.invoke(kwargs)
    logging.info(f"{result=}")
    return result.content


def invoke_gemini_search(prompt, system_instruction=None):
    google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = gemini_search_models[0]

    google_search_tool = Tool(google_search=GoogleSearch())
    response = google_client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            system_instruction=system_instruction,
            response_modalities=["TEXT"],
        ),
    )

    parts = response.candidates[0].content.parts
    output = [each.text for each in parts]
    return output


def invoke_gemini_search_json(prompt, system_instruction=None):
    """Invoke Gemini with Google Search tool and return parsed JSON.

    Expects the model to return a JSON object as plain text. If parsing fails,
    attempts to extract the first JSON object from the text.
    """
    text = "".join(invoke_gemini_search(prompt, system_instruction))

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                logging.warning("invoke_gemini_search_json: JSON extraction failed")
        logging.warning("invoke_gemini_search_json: returning raw text due to parse failure")
        return {"raw": text}
