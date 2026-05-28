import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d \t %(message)s"
)
logger = logging.getLogger(__name__)

logger.info('Importando bibliotecas e módulos necessários...')
import httpx
import json

from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query
from typing import List
from starlette.middleware.cors import CORSMiddleware
from doxxo.conteudo.banco_vetorial import BancoVetorial

logger.info('Carregando banco vetorial e preparando coleções...')
banco_vetorial = BancoVetorial(
    url_banco_vetorial='../banco_vetorial'
)

colecoes = defaultdict(list)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando gerador de embeddings...")
    gerador_embeddings = banco_vetorial.carregar_gerador_embeddings(
        nome_modelo='gemini-embedding-2'
    )

    logger.info("Carregando interface ChromaDB...")
    for nome_colecao in banco_vetorial.listar_nomes_colecoes():
        colecoes[nome_colecao] = banco_vetorial.conectar_colecao_documentos(
            url_banco='../banco_vetorial',
            nome_colecao=nome_colecao,
            gerador_embeddings=gerador_embeddings,
            criar_colecao_automaticamente=False
        )

    cliente_http = httpx.AsyncClient(timeout=60.0)

    app.state.gerador_embeddings = gerador_embeddings
    app.state.colecoes_documentos = colecoes
    app.state.cliente_http = cliente_http

    logger.info("API inicializada")

    yield

    # Exemplo de código de SHUTDOWN (se necessário)
    logger.info("Desligando e liberando recursos...")
    await cliente_http.aclose()
    if gerador_embeddings and hasattr(gerador_embeddings.modelo, 'to'):
        gerador_embeddings.modelo.to('cpu') 
        del gerador_embeddings.modelo
    logger.info("Recursos liberados com sucesso.")

logger.info('Instanciando a api (FastAPI)...')
controller = FastAPI(lifespan=lifespan)

controller.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@controller.get('/doxxo/health')
async def chat_health():
    return {"status": "ok"}

@controller.get('/doxxo/consulta')
async def consultar_documentos(
    request: Request,
    pergunta: str,
    colecao: List[str]| None = Query(None),
    num_resultados: int = 5,
    filtros_metadados=None,
    filtros_texto=None):

    if colecao is None or len(colecao) == 0:
        colecoes = list(request.app.state.colecoes_documentos.keys())
    else:
        colecoes = colecao

    resultado_completo = []
    
    for colecao in colecoes:
        resultado = request.app.state.colecoes_documentos[colecao].consultar_documentos(
            termos_de_consulta=pergunta, 
            num_resultados=num_resultados,
            filtros_metadados=json.loads(filtros_metadados) if filtros_metadados else None,
            filtros_texto=json.loads(filtros_texto) if filtros_texto else None
        )

        ids = resultado["ids"][0]
        docs = resultado["documents"][0]
        metas = resultado["metadatas"][0]
        dists = resultado["distances"][0]

        for i in range(len(ids)):
            resultado_completo.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": {**metas[i], 'colecao': colecao},
                "distance": dists[i]
            })

    resultado_completo.sort(key=lambda x: x["distance"])

    return resultado_completo[:num_resultados]

@controller.get('/doxxo/listar-conteudo')
async def listar_conteudo():
    nomes_colecoes = list(controller.state.colecoes_documentos.keys())
    documentos = {}
    for nome in nomes_colecoes:
        colecao = controller.state.colecoes_documentos[nome]
        documentos[nome] = colecao.listar_titulos_documentos()
    return documentos

@controller.get('/doxxo/listar-colecoes')
async def listar_colecoes():
    return {'colecoes': list(controller.state.colecoes_documentos.keys())}