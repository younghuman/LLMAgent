from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
from expert_action_predictor import ExpertActionPredictor
from output_parser import AutoGPTAction
import numpy as np
import json
import torch
import re

DELIMITER = "%%%%"
class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        loop_limit: int = 100,
        init_obs: str = None,
        expert_predictor: ExpertActionPredictor = None
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.loop_limit = loop_limit
        self.init_obs = init_obs
        self.expert_predictor = expert_predictor

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            ai_name,
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
        )

    def run(self, goals: List[str]) -> str:
        # Interaction Loop
        loop_count = 0
        actions = []
        cur_obs, cur_info, result = None, None, None

        if self.init_obs:
            cur_obs, cur_info = self.init_obs.split(DELIMITER)
        while loop_count < self.loop_limit:
            user_input = (
                "Determine which next command to use, "
                "and respond using the JSON format specified above:"
            )
            # Discontinue if continuous limit is reached
            loop_count += 1
            loop_msg = f"loop number:{loop_count}"
            self.full_message_history.append(SystemMessage(content=loop_msg))
            if cur_obs and loop_count == 1:
                self.full_message_history.append(SystemMessage(content=cur_obs))
                #print(cur_obs)
            elif result:
                pass
                #print(result)
            
            if cur_obs and cur_info and self.expert_predictor:
                info = json.loads(cur_info)
                if 'image_feat' in info and info['image_feat'] is not None:
                   info['image_feat'] = torch.tensor(info['image_feat'])
                cur_obs = cur_obs.replace("=Observation=\n", "")
                # print("########", cur_obs, info)
                action = self.expert_predictor.predict(cur_obs, info)
                tool_name, tool_input = action.replace("]", "").split("[")
                action = f"{tool_name} with '{tool_input}'"
                user_input = f"Here's one suggestion for the command: {action}.\n" +\
                              "Please use your best judgement and feel free to disagree with the example. " + user_input
            
            print(loop_msg)
            print(user_input)
            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)

            if result:
                match = re.search(r"\{'reward':([\d\.]+)\}", result)
                if match:
                    action = AutoGPTAction(
                        name="finish",
                        args={"response": "I have successfully purchased the hair mask and completed all my objectives."}
                    )
            actions.append(action)
            cur_obs, cur_info = None, None
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                break
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                results = observation.split(DELIMITER)
                if len(results) == 2:
                    cur_obs, cur_info = results
                else:
                    cur_obs = results[0]
                result = f"Command {tool.name} returned: {cur_obs}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )
            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    break
                memory_to_add += feedback
            print(result)
            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))
        return self.full_message_history, actions