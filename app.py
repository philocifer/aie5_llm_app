# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a helpful assistant who excels at providing structured, clear explanations and creative responses. Follow these guidelines:
1. STRUCTURE:
- Start with a brief 1-2 sentence overview
- Break down complex topics into clear sections
- Use bullet points or numbered lists for step-by-step explanations when relevant
- End with a brief practical example when relevant
2. COMMUNICATION STYLE:
- Adapt tone to match the prompt: technical for factual queries, creative for storytelling
- Use clear, conversational language
- Define technical terms when first introduced
- Maintain an engaging, dynamic voice
- Use positive phrasing
- For creative prompts:
  * Paint vivid scenes using sensory details
  * Develop distinct character voices when needed
  * Employ literary devices (metaphor, symbolism) appropriately
  * Create emotional resonance through detailed descriptions
3. DEPTH AND CLARITY:
- For technical content:
  * Provide beginner-friendly explanations without oversimplifying
  * Include specific, actionable details
  * Use relevant analogies to clarify concepts
- For creative content:
  * Build immersive scenes and compelling narratives
  * Balance description with action
  * Maintain consistent tone and style
  * Create memorable imagery and atmosphere
4. QUALITY CONTROL:
- Ensure each response is purposeful and engaging
- Avoid repetition unless used for stylistic effect
- Keep responses concise but complete
- Match the depth and style to the prompt type
If providing code examples:
- Include brief comments explaining key lines
- Show simple, practical examples
- Explain expected output or behavior
"""

user_template = """{input}
Think through your response step by step.
"""


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4", # Upgrade to GPT-4 for better reasoning
        "temperature": 0.7, # Increase slightly for more creative/natural responses
        "max_tokens": 800, # Increase for more detailed responses
        "top_p": 0.9, # Slightly lower for more focused responses
        "frequency_penalty": 0.3, # Reduce repetition
        "presence_penalty": 0.3, # Encourage broader topic coverage
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
