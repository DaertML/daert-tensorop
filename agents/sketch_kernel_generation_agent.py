from ollama import chat

class SketchKernelGenerationAgent:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def generate_sketch_kernel(self, prompt):
        response = chat(model=self.model_name, messages=[{
            'role': 'user',
            'content': prompt,
        }])
        return response['message']['content']
