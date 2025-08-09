# ruff: noqa: B008

from components.vault_service.main import VaultService
from fastapi import Depends, FastAPI, HTTPException

from .models import (
    DocumentResponse,
    FileListResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
)


def create_app(service: VaultService) -> FastAPI:
    """
    Creates and configures the FastAPI application, registering all routes.
    This function returns the app object but does not run it.

    Args:
        service: The fully initialized VaultService instance.

    Returns:
        The configured FastAPI app instance.
    """
    app = FastAPI(title="Vault API")

    # Dependency provider to make the service available to endpoints
    def get_service() -> VaultService:
        return service

    # Register all API routes
    @app.get(
        "/files",
        response_model=FileListResponse,
        tags=["documents"],
        operation_id="list_files",
    )
    def list_files(svc: VaultService = Depends(get_service)) -> FileListResponse:
        files = svc.list_all_files()
        return FileListResponse(files=files, total_count=len(files))

    @app.get(
        "/document",
        response_model=DocumentResponse,
        tags=["documents"],
        operation_id="get_document",
    )
    def get_document(
        file_path: str, svc: VaultService = Depends(get_service)
    ) -> DocumentResponse:
        try:
            content = svc.get_document_content(file_path)
            return DocumentResponse(content=content, file_path=file_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="Document not found") from e

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["search"],
        operation_id="search_documents",
    )
    async def search(
        request: QueryRequest, svc: VaultService = Depends(get_service)
    ) -> QueryResponse:
        results = await svc.search_chunks(request.query, request.limit)
        return QueryResponse(sources=results)

    @app.post(
        "/reindex",
        response_model=ReindexResponse,
        tags=["admin"],
        operation_id="reindex_vault",
    )
    async def reindex(svc: VaultService = Depends(get_service)) -> ReindexResponse:
        result = await svc.reindex_vault()
        return ReindexResponse(**result)

    return app
