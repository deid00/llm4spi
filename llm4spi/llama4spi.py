
from llama_cpp import Llama
from datetime import datetime
from typing import Dict
import time
import os

from openai4spi import PromptResponder, generate_results

class MyLLAMA_Client(PromptResponder):
    """
    An instance of prompt-responder that uses a LLAMA as the backend model.
    """

    def __init__(self, client:Llama):
        PromptResponder.__init__(self)
        self.client = client

    def completeIt(self, multipleAnswer: int, prompt: str) -> list[str]:
        if self.DEBUG: print(">>> PROMPT:\n" + prompt)
        answers = []
        for k in range(multipleAnswer):
            # iterating inside the session does not work for various (open source) LLMs,
            # they keep giving the same answer despite the repeat-penalty
            #with self.client.chat_session():
            A = self.client(prompt,
                                      temperature=0.7,
                                      max_tokens=1024
                                      )
            answers.append(A["choices"][0]["text"].strip())
                # answer2 = self.client.generate("Please only give the Python code, without comment.", max_tokens=1024)
                # srtipping header seems difficult for some LLM :|
                # answer3 = self.client.generate("Please remove the function header.", max_tokens=1024)
            if self.DEBUG:
                print(f">>> raw response {k}:\n {A}")
        return answers

if __name__ == '__main__':
    llamaClient = Llama(model_path="/models/Nxcode-CQ-7B-orpo.gguf", n_gpu_layers=-1)

    myAIclient = MyLLAMA_Client(llamaClient)


    dataset = os.path.join( "/llm4spiDatasets/data/HEx-compact.json")

    generate_results(myAIclient,
                     dataset,
                     specificProblem =  None,
                     experimentName = "Nxcode-CQ-7B-orpo",
                     enableEvaluation=True,
                     allowMultipleAnswers=10,
                     prompt_type="usePredDesc"
                     #prompt_type="cot2"
                     )