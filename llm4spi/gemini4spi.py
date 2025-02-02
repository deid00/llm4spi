
from openai4spi import PromptResponder, generate_results
import google.generativeai as Gemini
import time

class MyGeminiResponder(PromptResponder):
    """
    An instance of prompt-responder that uses Google Gemini models.
    Free Gemini models have 3 limits
    RPM: requests per minute
    TPM: tokens per minute
    RPD: requests per day
    """
    def __init__(self, client:Gemini, rpm_limit, tmp_limit, rpd_limit):
        PromptResponder.__init__(self)
        self.client = client

        # to exceed num of request  per minute
        self.last_time_seen = -1
        # to set as parameters
        self.rpm_limit = rpm_limit
        self.tmp_limit = tmp_limit
        self.rpd_limit = rpd_limit
        # used resources
        self.rpm_used = 0
        self.tpm_used = 0
        self.rpd_used = 0
        # timers for minutes and days
        self.minute_timer = 0
        self.day_timer = 0



    def completeIt(self, multipleAnswer: int, prompt: str) -> list[str]:

        if self.DEBUG: print(">>> PROMPT:\n" + prompt)
        answers = []

        # Gemini configuration
        cfg = Gemini.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024
        )

        if self.last_time_seen == -1 :
            self.last_time_seen = time.time()
            passed_time = 0
        else:
            passed_time = time.time() - self.last_time_seen

        # More than one minute is passed
        if passed_time > 60 :
            self.minute_timer = 0
            self.rpd_used = 0
            self.tpm_used = 0

        # One day is passed
        if passed_time > 60 * 60 * 24 :
            self.day_timer = 0
            self.rpd_used = 0

        # estimate token usage
        prompt_tokens = self.client.count_tokens(prompt)


        for k in range(multipleAnswer):
            print(f'####################### Time: {time.time():.0f}')
            print(f'####################### TPM: {self.rpm_used}')
            # rpm or tpm are finished
            if self.rpm_used + 1 >= self.rpm_limit or self.tpm_used + prompt_tokens.total_tokens + 1 >= self.tmp_limit :
                sleep_time = max(1,60 - self.minute_timer)
                if self.DEBUG: print(f">>> sleep {sleep_time}s\n" )
                time.sleep(sleep_time)
                self.rpm_used = 0
                self.tpm_used = 0
                self.minute_timer = 0

            # rpd finished
            if self.rpd_used >= self.rpd_limit :
                sleep_time = max(1, ( 60 * 60 * 14) - self.day_timer)
                if self.DEBUG: print(f">>> sleep {sleep_time}s\n")
                time.sleep(sleep_time)
                self.rpd_used = 0
                self.day_timer = 0

            t0 = time.time()
            response = self.client.generate_content(prompt, generation_config = cfg)
            t1 = time.time()
            self.minute_timer += (t1 - t0)
            self.day_timer += (t1 - t0)
            self.rpm_used += 1
            self.rpd_used += 1
            self.tpm_used += response.usage_metadata.total_token_count

            answers.append(response.text)
            # if self.DEBUG:
            #    print(f">>> raw response {k}:\n {response.text}")

        self.last_time_seen = time.time()
        return answers