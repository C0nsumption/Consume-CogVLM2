import argparse
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

class ImageAnalyzer:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_type,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    @staticmethod
    def apply_prompt_template(prompt):
        return (
            'A chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:".format(prompt)
        )

    def __call__(self, image, query, max_new_tokens=2048):
        history = []
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]]
        }

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "num_beams": 1,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in response else response.strip()

    def analyze_image(self, image_path, query, max_new_tokens=2048, save_response=False):
        raw_image = Image.open(image_path).convert('RGB')
        response = self.__call__(raw_image, query, max_new_tokens)
        print(f"==> {os.path.basename(image_path)}: {response}")

        if save_response:
            base_path, ext = os.path.splitext(image_path)
            response_path = f"{base_path}.txt"
            with open(response_path, "w") as f:
                f.write(response)
            print(f"Response saved to: {response_path}")

    def analyze_directory(self, directory_path, query, max_new_tokens=2048, save_response=False):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, filename)
                self.analyze_image(image_path, query, max_new_tokens, save_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Analysis using CogVLM2 Model')
    parser.add_argument('path', type=str, help='Path to the image file or directory containing images')
    parser.add_argument('query', type=str, help='Query to ask about the image(s)')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Maximum number of new tokens to generate')
    parser.add_argument('--save_response', action='store_true', help='Save the response to a text file')
    args = parser.parse_args()

    analyzer = ImageAnalyzer("./cogvlm2-llama3-chat-19B-int4")
    
    if os.path.isdir(args.path):
        analyzer.analyze_directory(args.path, args.query, args.max_new_tokens, args.save_response)
    else:
        analyzer.analyze_image(args.path, args.query, args.max_new_tokens, args.save_response)
