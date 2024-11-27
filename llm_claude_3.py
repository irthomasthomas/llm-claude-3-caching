from anthropic import Anthropic, AsyncAnthropic
import llm
import click
from pydantic import Field, field_validator, model_validator
from typing import Optional, List
import os
import json
from pathlib import Path

# Add these helper functions at the top of the file after imports
def get_config_path():
    config_dir = Path(click.get_app_dir("llm"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "claude_cache_config.json"

def load_cache_config():
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        "CLAUDE_CACHE_ALL_PROMPT": "0",
        "CLAUDE_CACHE_USER_PROMPT": "0", 
        "CLAUDE_CACHE_SYSTEM_PROMPT": "0"
    }

def save_cache_config(config):
    with open(get_config_path(), "w") as f:
        json.dump(config, f)
        
def set_env_from_config():
    config = load_cache_config()
    for key, value in config.items():
        os.environ[key] = value

# Call this when module loads
set_env_from_config()

@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def cache():
        """ Commands for managing prompt caching """
        pass
    
    @cache.command()
    @click.argument('prompt_type', type=click.Choice(['user', 'system', 'all']))
    def enable(prompt_type):
        """Enable caching for specified prompt type"""
        config = load_cache_config()
        config[f'CLAUDE_CACHE_{prompt_type.upper()}_PROMPT'] = '1'
        save_cache_config(config)
        set_env_from_config()
        click.echo(f'{prompt_type.capitalize()} prompt caching enabled')

    @cache.command()
    @click.argument('prompt_type', type=click.Choice(['user', 'system', 'all']))
    def disable(prompt_type):
        """Disable caching for specified prompt type"""
        config = load_cache_config()
        config[f'CLAUDE_CACHE_{prompt_type.upper()}_PROMPT'] = '0'
        save_cache_config(config)
        set_env_from_config()
        click.echo(f'{prompt_type.capitalize()} prompt caching disabled')

    @cache.command()
    def status():
        """Check current caching status"""
        config = load_cache_config()
        user_cache = config['CLAUDE_CACHE_USER_PROMPT'] == '1'
        system_cache = config['CLAUDE_CACHE_SYSTEM_PROMPT'] == '1'
        all_cache = config['CLAUDE_CACHE_ALL_PROMPT'] == '1'
        if all_cache:
            click.echo('All prompts caching: Enabled')
        else:
            click.echo(f'User prompt caching: {"Enabled" if user_cache else "Disabled"}')
            click.echo(f'System prompt caching: {"Enabled" if system_cache else "Disabled"}')

@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    ),
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
    cache_prompt: Optional[bool] = Field(
        description="Whether to cache the user prompt for future use",
        default=None,
    )
    
    cache_system: Optional[bool] = Field(
        description="Whether to cache the system prompt for future use",
        default=None,
    )
    
    @field_validator("cache_prompt", "cache_system")
    @classmethod
    def validate_cache_settings(cls, value, info):
        if value is not None:
            return value
            
        config = load_cache_config()
        all_cache = config.get('CLAUDE_CACHE_ALL_PROMPT') == '1'
        if all_cache:
            return True
            
        if info.field_name == "cache_prompt":
            return config.get('CLAUDE_CACHE_USER_PROMPT') == '1'
        else:
            return config.get('CLAUDE_CACHE_SYSTEM_PROMPT') == '1'

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
    )

    top_p: Optional[float] = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        real_max = cls.model_fields["max_tokens"].default
        if not (0 < max_tokens <= real_max):
            raise ValueError("max_tokens must be in range 1-{}".format(real_max))
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
    def validate_model(self):
        # Existing temperature/top_p validation
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
            
        # Validate cache settings are consistent with config
        config = load_cache_config()
        if config.get('CLAUDE_CACHE_ALL_PROMPT') == '1':
            self.cache_prompt = True
            self.cache_system = True
            
        return self


long_field = Field(
    description="The maximum number of tokens to generate before stopping",
    default=4_096 * 2,
)


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
            self.extra_headers={"anthropic-beta": "pdfs-2024-09-25", "anthropic-beta": "prompt-caching-2024-07-31"}
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

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        cache_control_count = 0
        max_cache_control_blocks = 2  # Leave one for the current prompt and system prompt
        
        if conversation:
            for response in reversed(conversation.responses):
                content = []
                
                if response.attachments:
                    for attachment in response.attachments:
                        content.append({
                            "type": "document" if attachment.resolve_type() == "application/pdf" else "image",
                            "source": {
                                "data": attachment.base64_content(),
                                "media_type": attachment.resolve_type(),
                                "type": "base64",
                            },
                        })
                
                text_content = {
                    "type": "text",
                    "text": response.prompt.prompt
                }
                
                if prompt.options.cache_prompt is not False and cache_control_count < max_cache_control_blocks:
                    text_content["cache_control"] = {"type": "ephemeral"}
                    cache_control_count += 1
                    
                content.append(text_content)

                messages.extend([
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": response.text()}
                ])

        if prompt.attachments:
            content = []
            for attachment in prompt.attachments:
                content.append({
                    "type": "document" if attachment.resolve_type() == "application/pdf" else "image",
                    "source": {
                        "data": attachment.base64_content(),
                        "media_type": attachment.resolve_type(),
                        "type": "base64",
                    }
                })
            
            text_content = {
                "type": "text", 
                "text": prompt.prompt
            }
            if prompt.options.cache_prompt:
                text_content["cache_control"] = {"type": "ephemeral"}
            content.append(text_content)
            
            messages.append({"role": "user", "content": content})
        else:
            text_content = {
                "type": "text",
                "text": prompt.prompt
            }
            if prompt.options.cache_prompt:
                text_content["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "user", "content": [text_content]})
            
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
            
        if prompt.system:
            kwargs["system"] = [{
                "type": "text",
                "text": prompt.system,
                "cache_control": {"type": "ephemeral"} if prompt.options.cache_system else None
            }]
            
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        return kwargs

    def set_usage(self, response):
        usage = response.response_json.get("usage")
        if usage:
            response.set_usage(
                input=usage.get("input_tokens"), output=usage.get("output_tokens")
            )

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessages(_Shared, llm.Model):

    def execute(self, prompt, stream, response, conversation):
        client = Anthropic(api_key=self.get_key(), default_headers={"anthropic-beta": "prompt-caching-2024-07-31"})
        kwargs = self.build_kwargs(prompt, conversation)

        if stream:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.beta.prompt_caching.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()
        self.set_usage(response)


class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field


class AsyncClaudeMessages(_Shared, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = AsyncAnthropic(api_key=self.get_key(), default_headers={"anthropic-beta": "prompt-caching-2024-07-31"})
        kwargs = self.build_kwargs(prompt, conversation)
        if stream:
            async with client.messages.stream(**kwargs) as stream_obj:
                async for text in stream_obj.text_stream:
                    yield text
            response.response_json = (await stream_obj.get_final_message()).model_dump()
        else:
            completion = await client.beta.prompt_caching.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()
        self.set_usage(response)


class AsyncClaudeMessagesLong(AsyncClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field
