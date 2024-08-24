from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

app = FastAPI()

model_name = "microsoft/Phi-3.5-mini-instruct"  # 사용할 모델 이름
tokenizer_name = "microsoft/Phi-3.5-mini-instruct"  # 모델에 맞는 토크나이저 이름

model = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

alpaca_prompt = """
### Instruction:
{}

### Context:
{}

### Input:
{}

### Response:
{}
"""

instruction = '''
다음은 작업을 설명하는 지시사항과 추가적인 맥락을 제공하는 입력이 쌍을 이루는 형태입니다.
단계별로 차근 차근 생각해서 결론 위주로 자세히 대답해줘.
한줄에 띄어쓰기 포함해서 60글자 이내로 써주고 그 이상일 때는, 줄바꿈을 해줘.
'''

def vectorize(text):
    with torch.no_grad():
        encoded = embed_model.encode(text, batch_size=60, max_length=8192)
    return encoded['dense_vecs']

def get_answer_from_vectordb(text, search_len):
    search_vectors = vectorize(text)
    output_fields = ["file_name", "div", "kwan", "jo", "text"]
    search_params = {"metric_type": "COSINE"} 
    results = collection.search([search_vectors], 
        "embedding", 
        search_params, 
        output_fields=output_fields, 
        limit=search_len)[0]


    context = ''
    for res in results:
        context += res.entity.div + " : " + res.entity.kwan + " : " + res.entity.jo + " : " + res.entity.text + '\n\n'
        
    context = context.replace(' \n', '')
    print(content)
    
    return context

def runner(model_path, token_len, thread_len, prompt):
    inputs = tokenizer(input_text, return_tensors="pt")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=token_len,
        do_sample=False
    )
    
    # 결과를 스트리밍 방식으로 생성
    for token in model.generate(inputs['input_ids'], sampling_params=sampling_params):
        yield tokenizer.decode(token, skip_special_tokens=True)

@app.get("/generate")
async def generate(
    input_text: str = Query(..., description="The input text for the model"),
    token_len: int = Query(50, description="The maximum number of tokens to generate")
):
    # Stream the response
    def stream_response():
        start_time = time.time()
        stream = generate_stream(input_text, token_len)
        for part in stream:
            yield part
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.6f} seconds")

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
