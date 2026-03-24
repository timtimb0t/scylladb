# Vector index in Scylla

Vector indexes are custom indexes (USING 'vector\_index'). Their `target` option in `system_schema.indexes` uses following format:

- Simple single-column vector index `(v)`: just the (escaped) column name, e.g. `v`
- Vector index with filtering columns `(v, f1, f2)`: JSON with `tc` (target column) and `fc` (filtering columns): `{"tc":"v","fc":["f1","f2"]}`
- Local vector index `((p1, p2), v)`: JSON with `tc` and `pk` (partition key columns): `{"tc":"v","pk":["p1","p2"]}`
- Local vector index with filtering columns `((p1, p2), v, f1, f2)`: JSON with `tc`, `pk`, and `fc`: `{"tc":"v","pk":["p1","p2"],"fc":["f1","f2"]}`

The `target` option acts as the interface for the vector-store service, providing the metadata necessary to determine which columns are indexed and how they are structured.
