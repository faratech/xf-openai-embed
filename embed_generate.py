#!/usr/bin/env python3
"""
embed_generate.py: Quick utility to generate an OpenAI embedding for sample text.
"""

import sys
import asyncio
from xf_embed.embeddings import generate_embedding
from xf_embed.config import settings


async def main():
    if len(sys.argv) < 2:
        print("Usage: python embed_generate.py '<your text here>'")
        return

    input_text = " ".join(sys.argv[1:])
    print(f"Generating embedding using '{settings.OPENAI_EMBEDDING_MODEL}'...")

    try:
        embedding = await generate_embedding(input_text, normalize=True)
        print(f"Embedding Dimensions: {len(embedding)}")
        print(f"First 5 floats: {embedding[:5].tolist()}")
        print(f"L2 Norm: {float((embedding**2).sum()) ** 0.5:.4f}")
    except Exception as e:
        print(f"Failed to generate embedding: {e}")


if __name__ == "__main__":
    asyncio.run(main())
