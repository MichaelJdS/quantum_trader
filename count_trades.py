import asyncio
import os
import sys

sys.path.insert(0, '.')
from infra.db.database import get_session, init_db
from infra.db.models_db import TradeModel
from sqlalchemy import select, func

async def main():
    await init_db()
    async with get_session() as session:
        res = await session.execute(select(func.count(TradeModel.id)))
        count = res.scalar()
        print(f"Total de trades no banco: {count}")

if __name__ == "__main__":
    asyncio.run(main())
