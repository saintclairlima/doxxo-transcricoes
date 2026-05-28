import os
from dotenv import load_dotenv
from typing import List
from threading import Lock
import logging

from chromadb import PersistentClient, EmbeddingFunction

from doxxo.conteudo.colecao_documentos import ColecaoDocumentosChromaDB
from doxxo.conteudo.gerador_embedding import GeradorEmbeddingsGemini
from doxxo.processamento_documentos.models import Fragmento

logger = logging.getLogger(__name__)

load_dotenv()

class BancoVetorial:
    '''
    Classe para gerenciar múltiplas interfaces de bancos vetoriais

    Atributos:
        bancos_vetoriais (dict): dicionário com as interfaces dos bancos vetoriais, indexadas por nome
    '''
    def __init__(self, url_banco_vetorial):
        self.url_banco_vetorial = url_banco_vetorial
        self.geradores_embeddings = {}
        self.lock = Lock()

        self.banco_vetorial = PersistentClient(path=self.url_banco_vetorial)

    def carregar_gerador_embeddings(self, nome_modelo: str) -> EmbeddingFunction:
        '''
        Cria funções de embeddings com base em modelo pré-treinado informado

        Parâmetros:
            nome_modelo (str): nome do modelo a ser utilizado para geração de embeddings
            classe_modelo: classe do modelo a ser utilizado para geração de embeddings
            device (str): dispositivo onde o modelo será carregado ('cpu' ou 'cuda')
            url_cache_modelos (str): caminho para a pasta de cache dos modelos
            instrucao (str, opcional): instrução a ser utilizada na geração de embeddings (padrão: None)
        Retorna: 
            (FuncaoEmbeddings): função de embeddings criada
        '''

        # como o gerador de embeddings pode ser reutilizado em várias coleções,
        # mantém controle do que já foi criado para uso em outras coleções
        with self.lock: # (o lock evita condições de concorrência em situações multi-thread)
            self.geradores_embeddings[nome_modelo] = GeradorEmbeddingsGemini(
                nome_modelo=nome_modelo,
                chave_api=os.environ.get("GEMINI_API_KEY")
            )
        
        return self.geradores_embeddings[nome_modelo]
    
    def listar_nomes_colecoes(self) -> List[str]:
        '''
        Lista os nomes das coleções disponíveis no banco vetorial

        Retorna:
            (List[str]): lista com os nomes das coleções disponíveis
        '''
        
        colecoes = self.banco_vetorial.list_collections()
        nomes_colecoes = [colecao.name for colecao in colecoes]
        return nomes_colecoes
    
    def conectar_colecao_documentos(self, url_banco, nome_colecao, gerador_embeddings=None, criar_colecao_automaticamente: bool=False,) -> ColecaoDocumentosChromaDB:
        '''
        Obtém conexão com uma coleção de documentos de um banco vetorial ChromaDB

        Parâmetros:
            nome_banco (str): nome do banco vetorial
            nome_colecao (str): nome da coleção dentro do banco vetorial
            gerador_embeddings (FuncaoEmbeddings, opcional): função de embeddings a ser utilizada na interface (padrão: None)
        
        Retorna:
            (ColecaoDocumentosChromaDB): coleção do banco vetorial solicitado
        '''

        colecao_documentos = ColecaoDocumentosChromaDB(
            url_banco_vetorial=url_banco,
            nome_colecao=nome_colecao,
            gerador_embeddings=gerador_embeddings,
            criar_colecao_automaticamente=criar_colecao_automaticamente
        )
        
        return colecao_documentos    
    
    def criar_nova_colecao(self,
                           nome_colecao: str,
                           documentos: List[Fragmento] | None=None,
                           gerador_embeddings: EmbeddingFunction | None=None,
                           hsnw_space: str | None='cosine') -> ColecaoDocumentosChromaDB:

        if ColecaoDocumentosChromaDB.existe(url_banco_vetorial=self.url_banco_vetorial, nome_colecao=nome_colecao):
            raise Exception(f'A coleção "{nome_colecao}" já existe no banco vetorial "{self.url_banco_vetorial}"')
        
        argumentos_colecao = {
            'url_banco_vetorial': self.url_banco_vetorial,
            'nome_colecao': nome_colecao,
            'criar_colecao_automaticamente': True
        }
        if gerador_embeddings is not None:
            argumentos_colecao['gerador_embeddings'] = gerador_embeddings
        if hsnw_space is not None:
            argumentos_colecao['hnsw_space'] = hsnw_space
        
        colecao = ColecaoDocumentosChromaDB(**argumentos_colecao)

        logger.info(f'Coleção "{nome_colecao}" criada no banco vetorial "{self.url_banco_vetorial}"')

        if documentos:
            logger.info(f'Adicionando {len(documentos)} documentos à nova coleção "{nome_colecao}"...')

            # É possível adicionar todos os documentos em um único lote, mas é mais lento sem GPU e consome mais RAM
            # colecao.adicionar_documentos_colecao(documentos=documentos)

            # Adicionar individualmente sai mais viável em ambientes limitados
            for idx, documento in enumerate(documentos):
                colecao.adicionar_documentos_colecao(documentos=[documento])
                logger.info(f'Adicionado documento {idx + 1} de {len(documentos)} à nova coleção "{nome_colecao}"...')

        return colecao
    
    def remover_colecao(self, nome_colecao: str) -> None:
        '''
        Remove uma coleção do banco vetorial

        Parâmetros:
            nome_colecao (str): nome da coleção a ser removida
        '''

        self.banco_vetorial.delete_collection(name=nome_colecao)
        logger.info(f'Coleção "{nome_colecao}" removida do banco vetorial "{self.url_banco_vetorial}"')