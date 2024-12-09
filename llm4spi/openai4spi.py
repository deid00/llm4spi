#
# Provide a main API for evaluating an LLM on its ability to produce pre-/post-conditions.
# An example is also shown on how to use the API using openAI's GPT as the back-end LLM.
#

from datetime import datetime
from typing import Dict, List
from openai import OpenAI
import os
import time

from data import ZEROSHOT_DATA, read_problems, write_jsonl
from prompting import create_prompt
from evaluation import evaluate_tasks_results
from pythonSrcUtils import extractFunctionBody, extractPythonFunctionDef_fromMarkDownQuote, fix_indentation

class PromptResponder:

    def __init__(self) :
        self.DEBUG = False

    """
    A template class that generically represents an LLM/AI that can respond to a prompt 
    to ask its completion.
    """
    def completeIt(self, multipleAnswer:int, prompt:str) -> list[str]:
        """
        Complete the given prompt. Return the answer. As some LLMs can be configured to
        generate multiple answers to the same prompt, the parameter multipleAnswer can be used
        to specify how many asnwers we want from the AI. If its one, the just one answer is 
        expected from the AI. If it is e.g. 3 then three answers are expected. 
        """
        return None

def generate_results(
        AI : PromptResponder, 
        datafile:str,
        specificProblem:str,
        experimentName:str,
        enableEvaluation: bool,
        allowMultipleAnswers: int,
        prompt_type: str        
        ) -> None:
    """
    The general API for evaluating an LLM/AI in its ability to construct pre- and post-conditions
    from their informal descriptions. The AI is generically represented by an PromptResponder-object,
    which has a method takes a prompt-string as input, and returns a new string as the answer 
    to the prompt.
    
    This API takes a dataset as its input, which is a file containing a JSON-list.
    Each item in the list is called a Problem. It represents a program (for now, a Python-function) whose 
    pre- and post-conditions are to be constructed. The program-code and its doc are provided, but the 
    dataset is assumed to already contain separate text-doc for the pre- and post-conditions of the program. 

    If the parameter specificProblem is specified (it is not None), then only the problem with the
    specified id will be evaluated. So, the dataset then is just a singleton-set containing that
    single problem. 
    
    For each Problem in the dataset, the LLM task is to generate a python code that is an executable version 
    of the corresponding pre-/post-condition.
    The dataset is also assumed to contain a reference solution (ground truth) for each pre-/post-condition, 
    along with tests for assessing how good the answer from the AI. 

    For each pre- or post-condition R produced by the AI, the following evaluation result is produced:
       * (1) failed: if R crashes, or if it does not even return a boolean value, or if R
                     returns None on all test-cases.
       * (2) accepted: for every test-input x (provided in the dataset), R(x) = R0(x), where R0 is the provided
                 reference solution. Test inputs for which R(x) gives None is interpreted as 'i don't know' and
                 are ignored from the consideration.
       * (3) too strong: if R is not accepted, and for every test input x, for which R(x) is not None we have
                R(x) implies R0(x).
       * (4) too weak:   if R is not accepted, and for every test input x, for which R(x) is not None we have
                R0(x) implies R(x).
       * (5) rejected: none of the above judgement is the case.

    An evaluation report, along with the produced solutions from the AI are saved in files in /results.
    """
    time0 = time.time()
    tasks = read_problems(datafile)
    timeSpentReadingData = time.time() - time0

    if specificProblem != None:
        tasks = { specificProblem : tasks[specificProblem] }

    time1 = time.time()
    for task in tasks:
        generate_task_result(AI, tasks[task], allowMultipleAnswers, prompt_type=prompt_type)
    timeSpentAI = time.time() - time1

    current_date = (datetime.now()).strftime("%d_%m_%Y_%H_%M_%S")

    time2 = time.time()
    if enableEvaluation:
        reportfile_basename = f"results/{experimentName}_evaluation_{prompt_type}_{current_date}"
        evaluate_tasks_results(tasks,reportfile_basename)
        results = [{
            "task_id": tasks[task]["task_id"],
            "pre_condition_prompt" : tasks[task]["pre_condition_prompt"],
            "pre_condition_raw_responses": tasks[task]["pre_condition_raw_responses"],
            "pre_condition_completions": tasks[task]["pre_condition_completions"],
            "pre_condition_evaluations": tasks[task]["pre_condition_evaluations"],
            "pre_condition_evaluation": tasks[task]["pre_condition_evaluation"],
            "post_condition_prompt" : tasks[task]["post_condition_prompt"],
            "post_condition_raw_responses": tasks[task]["post_condition_raw_responses"],
            "post_condition_completions": tasks[task]["post_condition_completions"],
            "post_condition_evaluations": tasks[task]["post_condition_evaluations"],
            "post_condition_evaluation": tasks[task]["post_condition_evaluation"]
            } for task in tasks]

    else:
        results = [{
            "task_id": tasks[task]["task_id"],
            "pre_condition_prompt" : tasks[task]["pre_condition_prompt"],
            "pre_condition_raw_responses": tasks[task]["pre_condition_raw_responses"],
            "pre_condition_completions": tasks[task]["pre_condition_completions"],
            "post_condition_prompt" : tasks[task]["post_condition_prompt"],
            "post_condition_raw_responses": tasks[task]["post_condition_raw_responses"],
            "post_condition_completions": tasks[task]["post_condition_completions"]
            } for task in tasks]
    timeSpentAnalysis = time.time() - time2

    # Saving raw responses and evaluation results in a json-file:
    write_jsonl(f"results/{experimentName}_model_responses_{prompt_type}_{current_date}.jsonl", results)

    overallTime = time.time() - time0

    print(f"   time loading data: {timeSpentReadingData}")
    print(f"   time AI: {timeSpentAI}")
    print(f"   time analysis: {timeSpentAnalysis}")
    print(f"   time all: {overallTime}")
    
    return


def fix_completionString(header:str, completion:str) -> str :
    """
    Try to fix the completion string sent by AI, e.g. by stripping of
    the function header (we will only ask it to return function bodies). 
    """
    if completion==None: return None
    completion = extractPythonFunctionDef_fromMarkDownQuote(completion)
    body1 = extractFunctionBody(completion)
    body2 = fix_indentation(header,body1)
    if body2 != None :
        return body2
    return body1

    
def generate_task_result(
        AI: PromptResponder,
        task: Dict,
        allowMultipleAnswers: int,
        prompt_type: str) -> Dict:
    """
    This function takes the desciption of a task/problem, represented as a dictionary.
    It then creates the completion prompt for the pre- and post-condition for the task. 
    The prompt is sent to an AI model and the answer (the completion)
    is collected. The answer is added into the task-dictionary.

    The AI is generically represented by an object of class PromptResponder, which has
    a method that takes a prompt-string and returns a string (the answer).

    The creation of the prompt is coded in the module Prompting. 
    """
    pre_condition_prompt = create_prompt(task, condition_type="pre", prompt_type=prompt_type)
    task["pre_condition_prompt"] = pre_condition_prompt
    task["pre_condition_raw_responses"] = None
    task["pre_condition_completions"]   = None
    if pre_condition_prompt != None:
        # note that this gives one or more answers, in a list:
        pre_completions = AI.completeIt(allowMultipleAnswers,pre_condition_prompt)
        task["pre_condition_raw_responses"] = pre_completions
        preCondHeader = task["pre_condition_incomplete"]
        task["pre_condition_completions"] = [ fix_completionString(preCondHeader,rawAnswer) for rawAnswer in pre_completions ]

    post_condition_prompt = create_prompt(task, condition_type="post", prompt_type=prompt_type)
    task["post_condition_prompt"] = post_condition_prompt
    task["post_condition_raw_responses"] = None
    task["post_condition_completions"] = None
    if post_condition_prompt != None:
        # note that this gives one or more answers, in a list:
        post_completions = AI.completeIt(allowMultipleAnswers,post_condition_prompt)
        task["post_condition_raw_responses"] = post_completions
        postCondHeader = task["post_condition_incomplete"]
        task["post_condition_completions"] = [ fix_completionString(postCondHeader,rawAnswer) for rawAnswer in post_completions ]

    return task

class MyOpenAIClient(PromptResponder):
    """
    An instance of prompt-responder that uses openAI LLM as the backend model.
    """
    def __init__(self,client: OpenAI, modelId:str):
        PromptResponder.__init__(self)
        self.client = client
        self.model = modelId 
    
    def completeIt(self, multipleAnswer:int, prompt:str) -> list[str] :
        if self.DEBUG: print(">>> PROMPT:\n" + prompt)
        completion = self.client.chat.completions.create(
            #model = "gpt-3.5-turbo",
            model = self.model,
            n = multipleAnswer,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ]
            )
        N = min(multipleAnswer, len(completion.choices))
        responses = [ completion.choices[k].message.content for k in range(N) ]
        if self.DEBUG: 
            for k in range(len(responses)):
                print(f">>> raw response {k}:\n {responses[k]}")
        return responses



if __name__ == '__main__':
    openai_api_key = os.environ.get('OPENAI_API_KEY') 
    openAIclient = OpenAI(api_key=openai_api_key)
    modelId = "gpt-3.5-turbo"
    #modelId ="gpt-4-turbo"
    #modelId ="o1-mini" 
    myAIclient = MyOpenAIClient(openAIclient,modelId)
    myAIclient.DEBUG = True

    dataset = ZEROSHOT_DATA
    ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(ROOT, "..", "..", "llm4spiDatasets", "data", "x.json")
    #dataset = os.path.join(ROOT, "..", "..", "llm4spiDatasets", "data", "simple-specs.json")

    generate_results(myAIclient,
                     dataset, 
                     specificProblem = None,
                     experimentName = "gpt3.5",     
                     enableEvaluation=True, 
                     allowMultipleAnswers=10,
                     prompt_type="usePrgDesc"
                     #prompt_type="cot2"
                     )
    
 
    