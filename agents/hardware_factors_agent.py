from ollama import chat

class HardwareFactorsAgent:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def generate_hardware_factors(self, prompt):
        response = chat(model=self.model_name, messages=[{
            'role': 'user',
            'content': prompt,
        }])
        return response['message']['content']
