import {open, type DB} from '@op-engineering/op-sqlite';

let db: DB | null = null;

export function initVectorDB() {
  db = open({name: 'ragbot.db'});

  // Regular table for chunk text
  db.execute(`
    CREATE TABLE IF NOT EXISTS chunks (
      id   INTEGER PRIMARY KEY AUTOINCREMENT,
      content TEXT NOT NULL
    )
  `);

  // sqlite-vec virtual table: 384 dims (all-MiniLM-L6-v2)
  db.execute(`
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
    USING vec0(embedding float[384])
  `);

  console.log('VectorDB ready');
}

export function insertChunkWithEmbedding(content: string, embedding: number[]) {
  if (!db) throw new Error('DB not initialised');

  const insertChunk = db.execute(
    'INSERT INTO chunks (content) VALUES (?)',
    [content],
  );
  const rowId = insertChunk.insertId!;

  // sqlite-vec expects a JSON array string for the vector
  const vecStr = JSON.stringify(embedding);
  db.execute(
    'INSERT INTO vec_chunks(rowid, embedding) VALUES (?, vec(?));',
    [rowId, vecStr],
  );
}

export function retrieveTopK(queryEmbedding: number[], k = 3): string[] {
  if (!db) throw new Error('DB not initialised');

  const vecStr = JSON.stringify(queryEmbedding);
  const results = db.execute(
    `
    SELECT c.content, v.distance
    FROM vec_chunks v
    JOIN chunks c ON c.id = v.rowid
    WHERE v.embedding MATCH vec(?)
    ORDER BY v.distance
    LIMIT ?
    `,
    [vecStr, k],
  );

  return (results.rows?._array ?? []).map((r: any) => r.content as string);
}

export function clearAllChunks() {
  if (!db) throw new Error('DB not initialised');
  db.execute('DELETE FROM chunks');
  db.execute('DELETE FROM vec_chunks');
}