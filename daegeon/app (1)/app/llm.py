from langchain_community.chat_models import ChatOllama

Ollama = ChatOllama(
                    model="llama-3-maal-8b:latest",
                    format="json",  # 입출력 형식을 JSON으로 설정합니다.
                    temperature=0,  # 샘플링 온도를 0으로 설정하여 결정론적인 출력을 생성합니다.
                )