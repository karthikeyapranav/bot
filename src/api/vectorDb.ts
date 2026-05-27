import {
  moveAssetsDatabase,
  open,
  type DB,
} from '@op-engineering/op-sqlite';

let db: DB | null = null;

export type RetrievedChunk = {
  id: number;
  text: string;
  source: string | null;
  page: number | null;
  distance: number;
};

const KNOWLEDGE_DB = 'knowledge.db';

function rowsOf<T>(result: any): T[] {
  return result.rows?._array ?? result.rows ?? [];
}

function tableExists(tableName: string) {
  if (!db) return false;
  const result = db.executeSync(
    "SELECT name FROM sqlite_schema WHERE type = 'table' AND name = ?",
    [tableName],
  );
  return rowsOf<{name: string}>(result).length > 0;
}

export async function initVectorDB() {
  const moved = await moveAssetsDatabase({
    filename: KNOWLEDGE_DB,
    overwrite: true,
  });

  if (!moved) {
    throw new Error(`Could not copy bundled ${KNOWLEDGE_DB} from assets/custom.`);
  }

  db = open({name: KNOWLEDGE_DB});

  if (!tableExists('chunks') || !tableExists('chunk_vecs')) {
    throw new Error(`${KNOWLEDGE_DB} is missing required RAG tables.`);
  }

  const chunkCount = getChunkCount();
  console.log(`Knowledge DB ready with ${chunkCount} chunks`);
  return chunkCount;
}

/**
 * Retrieve the top-K most similar chunks for a given query embedding.
 * The DB is treated as READ-ONLY during the app session — nothing is
 * ever written here at query time.  The knowledge.db was built offline
 * in Python and ships pre-indexed inside the APK.
 */
export function retrieveTopK(queryEmbedding: number[], k = 3): RetrievedChunk[] {
  if (!db) throw new Error('DB not initialised');

  const vecStr = JSON.stringify(queryEmbedding);
  const results = db.executeSync(
    `
    SELECT
      c.id,
      c.text,
      c.source,
      c.page,
      v.distance
    FROM chunk_vecs v
    JOIN chunks c ON c.id = v.chunk_id
    WHERE v.embedding MATCH ?
      AND k = ?
    ORDER BY v.distance
    `,
    [vecStr, k],
  );

  return rowsOf<RetrievedChunk>(results);
}

export function getChunkCount() {
  if (!db) return 0;
  const result = db.executeSync('SELECT COUNT(*) AS count FROM chunks');
  const rows = rowsOf<{count: number}>(result);
  return Number(rows[0]?.count ?? 0);
}