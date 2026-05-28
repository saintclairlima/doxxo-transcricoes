from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import logging

logger = logging.getLogger(__name__)
    
class GeradorEmbeddingsGemini(EmbeddingFunction):
    def __init__(self, nome_modelo: str, chave_api: str, fazer_log: bool=True, tamanho_lote: int=10) -> EmbeddingFunction:
        if not chave_api:
            raise ValueError(
                'A variável de ambiente GEMINI_API_KEY deve estar definida para usar o gerador de embeddings Gemini.'
            )
        self.nome_modelo = nome_modelo
        self.cliente = genai.Client(api_key=chave_api)
        self.fazer_log = fazer_log
        self.tamanho_lote = tamanho_lote

    def __call__(self, input: Documents) -> Embeddings:
        """
        Gera embeddings fatiando o input para respeitar os limites de contexto.
        """
        embeddings = []
        total_documentos = len(input)

        for i in range(0, total_documentos, self.tamanho_lote):
            lote_atual = input[i : i + self.tamanho_lote]
            
            if self.fazer_log:
                logger.info(f"Processando lote: itens {i} até {min(i + self.tamanho_lote, total_documentos)} de {total_documentos}...")

            result = self.cliente.models.embed_content(
                model=self.nome_modelo,
                contents=lote_atual
            )
            
            for embedding in result.embeddings:
                embeddings.append(embedding.values)

        return embeddings
