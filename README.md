# doxxo
Api consulta do banco vetorial com dados das transcrições embedados com Gemini-embeddings-2

# Setup
Instale os pacotes necessários
`pip install -r requirements.txt`

Crie um .env dentro de `src` e troque o valor de GEMINI_API_KEY por sua chave da api do Gemini

Vá para a pasta `src` e inicialize a API
`cd src`
`uvicorn doxxo.api.controller:controller --workers 1 --host 0.0.0.0`

# Exemplo de Chamada à API
`http://localhost:8000/doxxo/consulta?pergunta=teste&num_resultados=5`
