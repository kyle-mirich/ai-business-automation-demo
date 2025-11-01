"""
FastAPI Main Application
AI Business Automation Demo Platform Backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from api.routers import inventory, support, rag_chatbot, financial

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Business Automation API",
    description="Backend API for AI-powered business automation agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Next.js domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(financial.router, prefix="/api/financial", tags=["financial"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["inventory"])
app.include_router(support.router, prefix="/api/support", tags=["support"])
app.include_router(rag_chatbot.router, prefix="/api/rag", tags=["rag"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Business Automation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
