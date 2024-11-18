import json
from anthropic import Anthropic, AsyncAnthropic
import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import re
from dataclasses import dataclass
from anthropic.types import ContentBlock

@llm.hookimpl
def register_models(register):
    # Register all Claude-3 models
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    )
    register(
        ClaudeMessages("claude-3-opus-latest"),
        AsyncClaudeMessages("claude-3-opus-latest"),
        aliases=("claude-3-opus",),
    )
    register(
        ClaudeMessages("claude-3-sonnet-20240229"),
        AsyncClaudeMessages("claude-3-sonnet-20240229"),
        aliases=("claude-3-sonnet",),
    )
    register(
        ClaudeMessages("claude-3-haiku-20240307"),
        AsyncClaudeMessages("claude-3-haiku-20240307"),
        aliases=("claude-3-haiku",),
    )
    # 3.5 models
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20240620"),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20240620"),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
        aliases=("claude-3.5-sonnet-new",),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        AsyncClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        aliases=("claude-3.5-haiku",),
    )

class ClaudeOptions(llm.Options):
    stop_sequences: Optional[Union[List[str], str]] = Field(
        description="A list of sequences that, when generated, will cause the model to stop generating further tokens. Can be provided as a list of strings or a JSON-formatted string.",
        default=None,
    )
    prefill: Optional[str] = Field(
        description="Text to prefill the assistant's response. The model will continue from this text.",
        default=None,
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )
    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0.",
        default=1.0,
    )
    top_p: Optional[float] = Field(
        description="Use nucleus sampling. Recommended for advanced use cases only.",
        default=None,
    )
    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token.",
        default=None,
    )
    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )
    cache: Optional[bool] = Field(
        description="Whether to cache both system and user prompts",
        default=None,
    )
    cache_prompt: Optional[bool] = Field(
        description="Whether to cache the user prompt for future use",
        default=None,
    )
    cache_system: Optional[bool] = Field(
        description="Whether to cache the system prompt for future use",
        default=None,
    )

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, stop_sequences):
        if isinstance(stop_sequences, str):
            try:
                return json.loads(stop_sequences)
            except json.JSONDecodeError:
                raise ValueError("stop_sequences must be a valid JSON string representing a list of strings")
        return stop_sequences

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        real_max = cls.model_fields["max_tokens"].default
        if not (0 < max_tokens <= real_max):
            raise ValueError(f"max_tokens must be in range 1-{real_max}")
        return max_tokens

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in range 0.0-1.0")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        
        # Add validation for cache options
        if self.cache is not None:
            if self.cache_prompt is not None or self.cache_system is not None:
                raise ValueError("Cannot use 'cache' together with 'cache_prompt' or 'cache_system'")
            
        return self

class _Shared:
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        extra_headers=None,
        supports_images=True,
        supports_pdf=False,
    ):
        self.model_id = model_id
        self.claude_model_id = claude_model_id or model_id
        self.extra_headers = extra_headers or {}
        if supports_pdf:
            self.extra_headers["anthropic-beta"] = "pdfs-2024-09-25"
        self.attachment_types = set()
        if supports_images:
            self.attachment_types.update(
                {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }
            )
        if supports_pdf:
            self.attachment_types.add("application/pdf")

    def strip_assistant_prefix(self, text: str) -> str:
        """Strips "Assistant: " prefix if present."""
        pattern = r"^Assistant:\s+"
        return re.sub(pattern, "", text)

    def extract_message_content(self, response) -> str:
        """Extracts text content from Claude API message response."""
        text = ""
        try:
            content = response.content
            if isinstance(content, list):
                for block in content:
                    if block.type == "text":
                        if text:
                            text += "\n"
                        block_text = str(block.text)
                        text += block_text
            elif isinstance(content, str):
                text = content
        except (KeyError, TypeError, AttributeError):
            pass
        return self.strip_assistant_prefix(text)

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                if response.attachments:
                    content = [
                        {
                            "type": (
                                "document"
                                if attachment.resolve_type() == "application/pdf"
                                else "image"
                            ),
                            "source": {
                                "data": attachment.base64_content(),
                                "media_type": attachment.resolve_type(),
                                "type": "base64",
                            },
                        }
                        for attachment in response.attachments
                    ]
                    content.append({"type": "text", "text": response.prompt.prompt})
                else:
                    content = response.prompt.prompt
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": content,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        if prompt.attachments:
            content = [
                {
                    "type": (
                        "document"
                        if attachment.resolve_type() == "application/pdf"
                        else "image"
                    ),
                    "source": {
                        "data": attachment.base64_content(),
                        "media_type": attachment.resolve_type(),
                        "type": "base64",
                    },
                }
                for attachment in prompt.attachments
            ]
            content.append({"type": "text", "text": prompt.prompt})
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_kwargs(self, prompt, conversation):
        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
            "max_tokens": prompt.options.max_tokens,
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        if prompt.system:
            kwargs["system"] = prompt.system

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        # Handle the new cache option
        if prompt.options.cache is not None:
            kwargs["prompt_caching"] = {
                "cache_user_prompt": prompt.options.cache,
                "cache_system_prompt": prompt.options.cache,
            }
        elif prompt.options.cache_prompt is not None or prompt.options.cache_system is not None:
            kwargs["prompt_caching"] = {
                "cache_user_prompt": prompt.options.cache_prompt,
                "cache_system_prompt": prompt.options.cache_system,
            }
            
        return kwargs

    def __str__(self):
        return f"Anthropic Messages: {self.model_id}"

class ClaudeMessages(_Shared, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        client = Anthropic(api_key=self.get_key(), default_headers={"anthropic-beta": "prompt-caching-2024-07-31"})
        kwargs = self.build_kwargs(prompt, conversation)
        if stream:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
            response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.beta.prompt_caching.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()

class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4_096 * 2,
        )

class AsyncClaudeMessages(_Shared, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = AsyncAnthropic(api_key=self.get_key())
        kwargs = self.build_kwargs(prompt, conversation)
        if stream:
            async with client.messages.stream(**kwargs) as stream_obj:
                async for text in stream_obj.text_stream:
                    yield text
            response.response_json = (await stream_obj.get_final_message()).model_dump()
        else:
            completion = await client.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()

class AsyncClaudeMessagesLong(AsyncClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4_096 * 2,
        )
