# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import anthropic
import requests
from requests.exceptions import ConnectionError
from tenacity import retry
from tenacity.wait import wait_random_exponential

from BackendBench.errors import AgentError

from .kernel_templates import KernelTemplateManager


class LLMKernelGenerator:
    """
    LLM Kernel Generator that uses direct Anthropic API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.model = model
        self.template_manager = KernelTemplateManager()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in environment or passed to constructor"
            )
        assert "claude" in self.model, "Only Claude (Anthropic) models are supported for now"

        self.client = anthropic.Anthropic(api_key=self.api_key)
        # check connection to the server
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.2,
                timeout=120.0,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
            )
        except anthropic.AnthropicError as e:
            raise ConnectionError(f"Cannot connect to Anthropic server: {e}")

    @property
    def readme_server_description(self) -> str:
        return "Direct Anthropic API"

    @property
    def readme_setup_section(self) -> str:
        return """## Setup
This backend uses the direct Anthropic API and requires:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```"""

    @retry(wait=wait_random_exponential(multiplier=2, min=1, max=60, exp_base=2))
    def call_llm(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.2,
                timeout=120.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            if not content:
                raise RuntimeError(
                    "API error: Empty response from LLM API (possible rate limit or outage)."
                )
            if "rate limit" in content.lower():
                raise RuntimeError("API error: Rate limit encountered from LLM API.")
            return content
        except anthropic.AnthropicError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"API error: Unexpected error: {e}")

    def generate_kernel(
        self,
        op_name: str,
        op_signature: str,
        op_description: str,
        framework: str = "triton",
        feedback: Optional[str] = None,
    ) -> str:
        if feedback:
            prompt = self.template_manager.create_refinement_prompt(
                op_name, op_signature, op_description, framework, feedback
            )
        else:
            prompt = self.template_manager.create_prompt(
                op_name, op_signature, op_description, framework
            )

        print("\n=== DEBUG: PROMPT SENT TO LLM RELAY ===")
        print(prompt)
        print("=== END PROMPT ===\n")

        try:
            content = self.call_llm(prompt)
            # Only raise AgentError if kernel extraction fails
            extracted_code = self._extract_code_from_response(content)

            print("\n=== DEBUG: RAW LLM RELAY RESPONSE ===")
            print(content)
            print("=== DEBUG: EXTRACTED CODE ===")
            print(extracted_code)
            print("=== END DEBUG ===\n")

            return extracted_code

        except Exception as e:
            raise RuntimeError(f"Agent error: Failed to generate kernel for {op_name}: {str(e)}")

    def _extract_code_from_response(self, response: str) -> str:
        if "```python" not in response:
            raise AgentError("Agent error: No Python code block found in LLM response.")
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        if end == -1:
            raise AgentError("Agent error: Unclosed Python code block in LLM response.")
        return response[start:end].strip()


class LLMRelayKernelGenerator(LLMKernelGenerator):
    """
    LLM Kernel Generator that uses local plugboard server.
    Inherits from LLMKernelGenerator and overrides call_llm method.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:11434",
        model: str = "gcp-claude-4-sonnet",
    ):
        self.server_url = server_url
        self.model = model
        self.template_manager = KernelTemplateManager()
        # Test connection to the server
        try:
            requests.get(f"{self.server_url}/", timeout=5)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to LLM relay server at {self.server_url}. ")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout connecting to LLM relay server at {self.server_url}. ")

    @property
    def readme_server_description(self) -> str:
        return "Local plugboard server (localhost:11434)"

    @property
    def readme_setup_section(self) -> str:
        return """## Server Setup
This backend requires the plugboard server to be running:
```
buck run @//mode/inplace run_plugboard_server -- --model gcp-claude-4-sonnet --pipeline usecase-dev-ai-user
```"""

    @retry(wait=wait_random_exponential(multiplier=2, min=1, max=60, exp_base=2))
    def call_llm(self, prompt: str) -> str:
        # Prepare request data for the plugboard server
        request_data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 8000,
            "top_p": 0.95,
        }

        # Bypass proxy for localhost connections
        proxies = (
            {"http": None, "https": None}
            if "127.0.0.1" in self.server_url or "localhost" in self.server_url
            else None
        )

        try:
            response = requests.post(
                self.server_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
                proxies=proxies,
            )
            if response.status_code != 200:
                raise ConnectionError(
                    f"Server returned status {response.status_code}: {response.text}"
                )
            response_data = response.json()
            content = response_data.get("output", "")
            if not content or "rate limit" in content.lower():
                raise RuntimeError("Empty response or rate limit encountered.")
            return content
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to communicate with LLM relay server: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in LLM relay call: {e}")
