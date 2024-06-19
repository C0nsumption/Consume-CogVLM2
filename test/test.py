import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./cogvlm2-llama3-chat-19B-int4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

def run_test_with_local_image():
    image_path = './assets/image.png'
    image = Image.open(image_path).convert('RGB')
    query = "Is there a banner in this image? If so what color is it and what does it say?"

    history = []

    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=history,
        images=[image],
        template_version='chat'
    )
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]]
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[1].strip()
        else:
            response = response.strip()
        print("\nCogVLM2:", response)

if __name__ == "__main__":
    run_test_with_local_image()
