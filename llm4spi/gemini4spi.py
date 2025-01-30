
from openai4spi import PromptResponder, generate_results
import google.generativeai as Gemini
import time

class MyGeminiResponder(PromptResponder):
    """
    An instance of prompt-responder that uses Google Gemini models.
    """
    def __init__(self, client:Gemini):
        PromptResponder.__init__(self)
        self.client = client
        # to keep track of time, towards deciding to pause so as not
        # to exceed num of request  per minute
        self.t0 = None

    def completeIt(self, multipleAnswer: int, prompt: str) -> list[str]:
        if self.DEBUG: print(">>> PROMPT:\n" + prompt)
        answers = []

        # Gemini configuration has for groq
        cfg = Gemini.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024
        )

        # Free Gemini API has token limits
        # To refine as they depend upon the model
        requestsPerMinuteLIMIT = 15
        requestPerDayLIMIT = 1500
        tokenPerMinuteLIMIT = 1000000
        if self.t0 == None:
            self.t0 = time.time()
        totRequestSinceLastPause = 0
        totTokensSinceLastPause = 0



        for k in range(multipleAnswer):
            try:
                response = self.client.generate_content(prompt, generation_config = cfg)
            except ValueError:
                print("Gemini exception")
                break

            answers.append(response.text)
            if self.DEBUG:
                print(f">>> raw response {k}:\n {response.text}")

            # account for API limits
            totTokensSinceLastPause = totTokensSinceLastPause + response.usage_metadata.total_token_count
            totRequestSinceLastPause = totRequestSinceLastPause + 1
            timeSinceLastPause = time.time() - self.t0

            if self.DEBUG:
                print(f">>> n request {k}: {totRequestSinceLastPause}")

            sleepTime = max(5, 4 - timeSinceLastPause)
            time.sleep(sleepTime)
            self.t0 = time.time()

            # if timeSinceLastPause > 60 :
            #     self.t0 =  time.time()
            #     totRequestSinceLastPause = 0
            #     totTokensSinceLastPause = 0
            # else:
            #     if multipleAnswer > 1 and ( totTokensSinceLastPause > tokenPerMinuteLIMIT * 0.8 or totRequestSinceLastPause > requestsPerMinuteLIMIT - 1 ):
            #         sleepTime = max(5, 60 - timeSinceLastPause)
            #         if self.DEBUG:
            #             print(f">>> SLEEPING {sleepTime}s ...")
            #         self.t0 = time.time()
            #         totRequestSinceLastPause = 0
            #         totTokensSinceLastPause = 0
            #         time.sleep(sleepTime)



        return answers